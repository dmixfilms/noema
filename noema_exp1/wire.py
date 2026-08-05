"""Formato de transmissão (wire format) do canal noético — Experimento 1.

Comprime o KV-cache de A em três eixos independentes e combináveis:
  - quantização: fp16 → int8 → int4 (escala simétrica por tensor)
  - janela temporal: só os primeiros SINK tokens (âncoras de atenção, à la
    StreamingLLM) + os últimos m tokens viajam
  - camadas: só as últimas k camadas viajam; as demais são zeradas no receptor

Os bytes medidos são o tamanho real do arquivo serializado. A reidratação
devolve (DynamicCache, pos_offset): pos_offset é a posição absoluta do próximo
token, para o RoPE de B continuar de onde A parou mesmo após o corte temporal.
"""
import torch
from transformers import DynamicCache

SINK = 4  # tokens-âncora do início mantidos em qualquer janela


def _quantizar(t: torch.Tensor, modo: str) -> dict:
    if modo == "fp16":
        return {"modo": modo, "q": t.to(torch.float16)}
    absmax = t.abs().amax()
    if modo == "int8":
        escala = (absmax / 127.0).clamp(min=1e-8)
        q = (t / escala).round().clamp(-127, 127).to(torch.int8)
        return {"modo": modo, "q": q, "escala": escala.to(torch.float32)}
    if modo == "int4":
        escala = (absmax / 7.0).clamp(min=1e-8)
        q = ((t / escala).round().clamp(-8, 7) + 8).to(torch.uint8)  # [0, 15]
        plano = q.flatten()
        if plano.numel() % 2:
            plano = torch.cat([plano, plano.new_zeros(1)])
        pares = plano[0::2] | (plano[1::2] << 4)  # 2 valores por byte
        return {"modo": modo, "q": pares, "forma": list(t.shape),
                "escala": escala.to(torch.float32)}
    raise ValueError(f"quantização desconhecida: {modo}")


def _dequantizar(d: dict, device: str, dtype: torch.dtype) -> torch.Tensor:
    modo = d["modo"]
    if modo == "fp16":
        return d["q"].to(device=device, dtype=dtype)
    if modo == "int8":
        return (d["q"].to(device).float() * d["escala"].to(device)).to(dtype)
    if modo == "int4":
        pares = d["q"].to(device)
        baixo = (pares & 0xF).float()
        alto = (pares >> 4).float()
        plano = torch.stack([baixo, alto], dim=-1).flatten()
        numel = 1
        for s in d["forma"]:
            numel *= s
        t = (plano[:numel] - 8.0) * d["escala"].to(device)
        return t.reshape(d["forma"]).to(dtype)
    raise ValueError(f"quantização desconhecida: {modo}")


def comprimir(k_list, v_list, quant="fp16", janela=None, camadas=None) -> dict:
    """Monta o pacote wire a partir das listas K/V por camada (tensores CPU)."""
    n_camadas = len(k_list)
    seq_len = k_list[0].shape[2]
    transmitidas = (list(range(n_camadas)) if not camadas
                    else list(range(n_camadas - camadas, n_camadas)))
    if janela and SINK + janela < seq_len:
        idx = list(range(SINK)) + list(range(seq_len - janela, seq_len))
    else:
        idx = None

    wire = {"n_camadas": n_camadas, "transmitidas": transmitidas,
            "pos_offset": seq_len, "k": [], "v": []}
    for i in transmitidas:
        k, v = k_list[i], v_list[i]
        if idx is not None:
            k, v = k[:, :, idx, :], v[:, :, idx, :]
        wire["k"].append(_quantizar(k, quant))
        wire["v"].append(_quantizar(v, quant))
    return wire


def reidratar(wire: dict, device: str, dtype: torch.dtype = torch.float16):
    """Reconstrói o cache no receptor. Camadas não transmitidas são zeradas
    (o custo dessa perda é justamente o que o experimento mede).
    Retorna (cache, pos_offset)."""
    transmitidas = {c: j for j, c in enumerate(wire["transmitidas"])}
    cache = DynamicCache()
    modelo_k = _dequantizar(wire["k"][0], device, dtype)
    modelo_v = _dequantizar(wire["v"][0], device, dtype)
    for i in range(wire["n_camadas"]):
        if i in transmitidas:
            j = transmitidas[i]
            k = _dequantizar(wire["k"][j], device, dtype)
            v = _dequantizar(wire["v"][j], device, dtype)
        else:
            k = torch.zeros_like(modelo_k)
            v = torch.zeros_like(modelo_v)
        cache.update(k, v, i)
    return cache, wire["pos_offset"]
