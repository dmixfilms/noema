"""Processo B do Experimento 1: conclui a partir de um cache COMPRIMIDO.

Para cada cache gravado pelo agente A do Exp 0:
  1. comprime no wire format pedido (quantização × janela × camadas),
  2. serializa em disco e mede os bytes reais,
  3. reidrata do disco e gera a resposta com o sufixo fixo.

Como no Exp 0, este processo nunca abre o arquivo de problemas.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noema_exp0"))
import config          # noqa: E402  (módulos do Exp 0)
import nucleo          # noqa: E402
import wire            # noqa: E402

DIR_EXP1 = Path(__file__).resolve().parent
DIR_RESULTADOS = DIR_EXP1 / "resultados"
DIR_WIRE = DIR_EXP1 / "wire_tmp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="nome da configuração")
    ap.add_argument("--quant", choices=["fp16", "int8", "int4"], default="fp16")
    ap.add_argument("--janela", type=int, default=0, help="últimos m tokens (0 = todos)")
    ap.add_argument("--camadas", type=int, default=0, help="últimas k camadas (0 = todas)")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    tok, _ = nucleo.carregar_modelo()
    DIR_RESULTADOS.mkdir(exist_ok=True)
    DIR_WIRE.mkdir(exist_ok=True)

    manifest = json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))
    ids_sufixo = tok(config.SUFIXO, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(config.DEVICE)

    linhas = []
    for m in manifest:
        payload = torch.load(config.DIR_CACHES / m["cache_file"],
                             map_location="cpu", weights_only=True)

        t0 = time.time()
        pacote = wire.comprimir(payload["k"], payload["v"], quant=args.quant,
                                janela=args.janela or None,
                                camadas=args.camadas or None)
        arq = DIR_WIRE / f"wire_{args.tag}_{m['id']}.pt"
        torch.save(pacote, arq)
        bytes_wire = arq.stat().st_size
        t_comprimir = time.time() - t0

        t0 = time.time()
        pacote_rx = torch.load(arq, map_location=config.DEVICE, weights_only=True)
        cache, pos_offset = wire.reidratar(pacote_rx, config.DEVICE, config.DTYPE)
        t_handoff = time.time() - t0
        arq.unlink()

        resposta, _ = nucleo.gerar_greedy(ids_sufixo, cache=cache,
                                          pos_offset=pos_offset)

        linhas.append({
            "id": m["id"],
            "config": args.tag,
            "resposta_gerada": resposta,
            "bytes_wire": bytes_wire,
            "bytes_cache_original": m["bytes_cache"],
            "latencia_comprimir_s": t_comprimir,
            "latencia_handoff_s": t_handoff,
        })
        print(f"[B1/{args.tag}] {m['id']}: {bytes_wire / 1e6:.2f} MB "
              f"({bytes_wire / m['bytes_cache'] * 100:.0f}% do original) "
              f"→ {resposta[:50]!r}", flush=True)

    saida = DIR_RESULTADOS / f"brutos_{args.tag}.jsonl"
    with open(saida, "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"[B1/{args.tag}] {len(linhas)} respostas gravadas em {saida}", flush=True)


if __name__ == "__main__":
    main()
