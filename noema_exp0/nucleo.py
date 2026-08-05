"""Núcleo do Experimento 0: geração manual token a token com DynamicCache.

Todo o controle do estado (KV-cache) passa por aqui. Sem model.generate():
cada token é um forward explícito, para que o cache possa ser cortado,
serializado e herdado por outro processo.
"""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import config

_tok = None
_model = None


def carregar_modelo():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(config.MODEL)
        # transformers >= 4.56 renomeou torch_dtype → dtype
        import transformers
        from packaging import version
        kw_dtype = ("dtype" if version.parse(transformers.__version__)
                    >= version.parse("4.56.0") else "torch_dtype")
        _model = AutoModelForCausalLM.from_pretrained(
            config.MODEL, device_map=config.DEVICE, **{kw_dtype: config.DTYPE}
        )
        _model.eval()
    return _tok, _model


def _mask(cache: DynamicCache, ids: torch.Tensor) -> torch.Tensor:
    # Máscara explícita cobrindo cache herdado + tokens novos (armadilha 2 do protocolo).
    return torch.ones(
        1, cache.get_seq_length() + ids.shape[1], dtype=torch.long, device=ids.device
    )


@torch.no_grad()
def _passo(model, ids: torch.Tensor, cache: DynamicCache):
    """Um forward sobre `ids` estendendo `cache`. Retorna (logits do último token, argmax)."""
    out = model(
        ids,
        past_key_values=cache,
        use_cache=True,
        attention_mask=_mask(cache, ids),
    )
    logits = out.logits[:, -1]
    return logits, logits.argmax(-1, keepdim=True)


@torch.no_grad()
def agente_A_pensa(problema: str, n_tokens_pensamento: int):
    """A processa o problema, raciocina e é cortado em N tokens.

    Retorna (cache, texto_parcial, concluiu_antes_do_corte, proximo_id_pendente).
    O cache contém prompt + N tokens de raciocínio já processados; o token
    pendente (argmax seguinte) NÃO está no cache — é descartado no handoff e
    usado apenas pela métrica de fidelidade (KL).
    """
    tok, model = carregar_modelo()
    msgs = [{"role": "user", "content": problema + "\nPense passo a passo."}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(config.DEVICE)

    cache = DynamicCache()
    _, next_id = _passo(model, ids, cache)

    gerados, concluiu = [], False
    for _ in range(n_tokens_pensamento):
        if next_id.item() == tok.eos_token_id:
            # A terminou sozinho antes do corte: flag p/ reporte separado (armadilha 6).
            concluiu = True
            break
        gerados.append(next_id)
        _, next_id = _passo(model, next_id, cache)

    texto_parcial = (
        tok.decode(torch.cat(gerados, dim=-1)[0], skip_special_tokens=True)
        if gerados
        else ""
    )
    return cache, texto_parcial, concluiu, next_id


@torch.no_grad()
def comprimento_cot_completo(problema: str, max_tokens: int) -> int:
    """Piloto de calibração: gera o CoT completo (greedy até EOS ou teto) e conta os tokens."""
    tok, model = carregar_modelo()
    msgs = [{"role": "user", "content": problema + "\nPense passo a passo."}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(config.DEVICE)

    cache = DynamicCache()
    _, next_id = _passo(model, ids, cache)
    n = 0
    for _ in range(max_tokens):
        if next_id.item() == tok.eos_token_id:
            break
        n += 1
        _, next_id = _passo(model, next_id, cache)
    return n


def _kv_do_cache(cache: DynamicCache):
    """Extrai as listas K/V por camada, compatível com as duas gerações da API de Cache
    (armadilha 7): `cache.layers[i].keys/.values` (recente) ou `key_cache/value_cache`."""
    layers = getattr(cache, "layers", None)
    if layers:
        try:
            return [l.keys for l in layers], [l.values for l in layers]
        except AttributeError:
            pass
    return list(cache.key_cache), list(cache.value_cache)


def serializar_cache(cache: DynamicCache, caminho: str) -> int:
    """Salva o estado mental de A em disco. Retorna bytes gravados."""
    k, v = _kv_do_cache(cache)
    torch.save({"k": [t.cpu() for t in k], "v": [t.cpu() for t in v]}, caminho)
    return os.path.getsize(caminho)


def carregar_cache(caminho: str) -> DynamicCache:
    """Reconstrói o cache no device via cache.update() — API estável entre versões."""
    payload = torch.load(caminho, map_location=config.DEVICE, weights_only=True)
    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(payload["k"], payload["v"])):
        cache.update(k.to(config.DEVICE), v.to(config.DEVICE), i)
    return cache


@torch.no_grad()
def gerar_greedy(ids: torch.Tensor, cache: DynamicCache | None = None,
                 max_new: int = config.MAX_NEW_B):
    """Continua a geração greedy a partir de `ids` (sobre um cache herdado, se houver).

    Retorna (texto, t_prefill_s): t_prefill é o tempo do primeiro forward — na
    condição T ele é o custo real do canal textual (B re-processa todo o contexto);
    na L é desprezível (só o sufixo).
    """
    tok, model = carregar_modelo()
    if cache is None:
        cache = DynamicCache()

    t0 = time.time()
    _, next_id = _passo(model, ids, cache)
    t_prefill = time.time() - t0

    resposta = []
    for _ in range(max_new):
        if next_id.item() == tok.eos_token_id:
            break
        resposta.append(next_id)
        _, next_id = _passo(model, next_id, cache)

    texto = (
        tok.decode(torch.cat(resposta, dim=-1)[0], skip_special_tokens=True)
        if resposta
        else ""
    )
    return texto, t_prefill
