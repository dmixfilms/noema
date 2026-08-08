"""Uma etapa da esteira, como processo independente — Experimento 4.

  --condicao L : herda o KV-cache da etapa anterior do disco e injeta apenas a
                 instrução de papel (o conteúdo do problema nunca é retransmitido)
  --condicao T : recebe o texto acumulado das etapas anteriores e re-processa
                 tudo do zero, como fazem os frameworks multi-agente de hoje

Uso: python agente_etapa.py --etapa 0|1|2 --condicao L|T
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent / "noema_exp0"))
sys.path.insert(0, str(RAIZ))

import config      # noqa: E402
import metrics     # noqa: E402
import nucleo      # noqa: E402
from esteira import ETAPAS  # noqa: E402

DIR_RESULTADOS = RAIZ / "resultados"
DIR_CACHES = RAIZ / "caches"


@torch.no_grad()
def _gerar(ids, cache, max_new):
    """Gera greedy a partir de `ids` sobre `cache`. Devolve texto, tempo de
    prefill (o custo de "entender o que chegou") e o cache resultante."""
    tok, model = nucleo.carregar_modelo()
    t0 = time.time()
    _, next_id = nucleo._passo(model, ids, cache)
    t_prefill = time.time() - t0

    gerados = []
    for _ in range(max_new):
        if next_id.item() == tok.eos_token_id:
            break
        gerados.append(next_id)
        _, next_id = nucleo._passo(model, next_id, cache)
    texto = (tok.decode(torch.cat(gerados, dim=-1)[0], skip_special_tokens=True)
             if gerados else "")
    return texto, t_prefill, cache


def _contexto_textual(tok, problema, saidas, etapa_idx):
    """Remonta em linguagem o que a etapa recebe — o jeito clássico: o agente
    lê o problema e o trabalho de todos os anteriores, do zero."""
    partes = [f"Problema: {problema}"]
    for i, texto in enumerate(saidas):
        partes.append(f"{ETAPAS[i + 1]['rotulo_textual']}: {texto}")
    conteudo = "\n\n".join(partes) + "\n\n" + ETAPAS[etapa_idx]["instrucao_papel"].strip()
    msgs = [{"role": "user", "content": conteudo}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt")
    # tokens de CONTEÚDO trafegados até esta etapa (fora a instrução fixa de papel)
    conteudo_trafegado = problema + "".join(saidas) if etapa_idx else ""
    n_conteudo = len(tok(conteudo_trafegado, add_special_tokens=False).input_ids)
    return ids.to(config.DEVICE), n_conteudo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", type=int, required=True)
    ap.add_argument("--condicao", choices=["L", "T"], required=True)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    tok, model = nucleo.carregar_modelo()
    # Aquecimento: o 1º forward de cada processo paga a inicialização dos
    # kernels CUDA (~200 ms) — fora do cronômetro p/ não poluir o prefill.
    with torch.no_grad():
        model(tok("aquecer", return_tensors="pt").input_ids.to(config.DEVICE))
    etapa = ETAPAS[args.etapa]
    max_new = etapa["max_tokens"] or config.MAX_NEW_B
    DIR_RESULTADOS.mkdir(exist_ok=True)
    DIR_CACHES.mkdir(exist_ok=True)

    problemas = metrics.ler_jsonl(config.ARQ_PROBLEMAS)
    n_max = int(os.environ.get("NOEMA_EXP4_N", "0"))
    if n_max:
        problemas = problemas[:n_max]
    anteriores = ([] if args.etapa == 0 else
                  [metrics.ler_jsonl(DIR_RESULTADOS / f"etapa{i}_{args.condicao}.jsonl")
                   for i in range(args.etapa)])

    linhas = []
    for j, p in enumerate(problemas):
        t_ini = time.time()

        if args.etapa == 0:
            # Porta de entrada: idêntica nas duas vias — o problema entra por
            # texto uma única vez (tokenização de entrada, não comunicação).
            msgs = [{"role": "user", "content": p["pergunta"] + "\n" +
                     etapa["instrucao_papel"]}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt").to(config.DEVICE)
            cache, n_conteudo, t_handoff = DynamicCache(), 0, 0.0
        elif args.condicao == "L":
            # Herda o pensamento da etapa anterior; só a instrução de papel viaja.
            t0 = time.time()
            cache = nucleo.carregar_cache(
                str(DIR_CACHES / f"cache_e{args.etapa - 1}_{p['id']}.pt"))
            t_handoff = time.time() - t0
            ids = tok(etapa["instrucao_papel"], return_tensors="pt",
                      add_special_tokens=False).input_ids.to(config.DEVICE)
            n_conteudo = 0
        else:
            # Via textual: relê problema + saídas anteriores, do zero.
            saidas = [a[j]["saida"] for a in anteriores]
            ids, n_conteudo = _contexto_textual(tok, p["pergunta"], saidas,
                                                args.etapa)
            cache, t_handoff = DynamicCache(), 0.0

        texto, t_prefill, cache = _gerar(ids, cache, max_new)

        bytes_cache = 0
        if args.condicao == "L" and args.etapa < len(ETAPAS) - 1:
            bytes_cache = nucleo.serializar_cache(
                cache, str(DIR_CACHES / f"cache_e{args.etapa}_{p['id']}.pt"))

        linhas.append({
            "id": p["id"], "etapa": args.etapa, "papel": etapa["nome"],
            "condicao": args.condicao, "saida": texto,
            "tokens_conteudo_recebidos": n_conteudo,
            "bytes_cache": bytes_cache,
            "latencia_handoff_s": t_handoff,
            "latencia_prefill_s": t_prefill,
            "latencia_etapa_s": time.time() - t_ini,
        })
        print(f"[e{args.etapa}/{args.condicao}] {p['id']}: {texto[:55]!r}", flush=True)

    saida = DIR_RESULTADOS / f"etapa{args.etapa}_{args.condicao}.jsonl"
    with open(saida, "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"[e{args.etapa}/{args.condicao}] {len(linhas)} linhas → {saida}", flush=True)


if __name__ == "__main__":
    main()
