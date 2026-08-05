"""Processo A: pensa até o corte e emite os dois canais de handoff.

Para cada problema:
  - canal latente  → serializa o KV-cache em caches/cache_problema_{i}.pt + manifest.json
  - canal textual  → registra o raciocínio parcial decodificado em handoff_texto.jsonl

Se --n-corte não for passado, calibra N com um piloto de N_PILOTO problemas
(N = CUT_RATIO × comprimento médio do CoT completo, com clamp).
"""
import argparse
import json
import statistics
import time

import torch

import config
import nucleo


def calibrar_n(problemas) -> int:
    piloto = problemas[: config.N_PILOTO]
    lens = []
    for p in piloto:
        n = nucleo.comprimento_cot_completo(p["pergunta"], config.MAX_COT_PILOTO)
        lens.append(n)
        print(f"[A/piloto] problema {p['id']}: CoT completo = {n} tokens", flush=True)
    media = statistics.mean(lens)
    n_corte = max(config.N_MIN, min(config.N_MAX, int(config.CUT_RATIO * media)))
    print(f"[A/piloto] média {media:.0f} tokens → N de corte = {n_corte}", flush=True)
    return n_corte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-corte", type=int, default=None,
                    help="corte fixo em tokens; se ausente, calibra via piloto")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    nucleo.carregar_modelo()

    problemas = [json.loads(l) for l in open(config.ARQ_PROBLEMAS, encoding="utf-8")]
    config.DIR_CACHES.mkdir(parents=True, exist_ok=True)

    n_corte = args.n_corte if args.n_corte is not None else calibrar_n(problemas)

    manifest, handoff_texto = [], []
    for p in problemas:
        t0 = time.time()
        cache, texto_parcial, concluiu, _ = nucleo.agente_A_pensa(p["pergunta"], n_corte)
        t_pensar = time.time() - t0

        arq = config.DIR_CACHES / f"cache_problema_{p['id']}.pt"
        t0 = time.time()
        nbytes = nucleo.serializar_cache(cache, str(arq))
        t_serializar = time.time() - t0

        manifest.append({
            "id": p["id"],
            "cache_file": arq.name,
            "n_corte": n_corte,
            "bytes_cache": nbytes,
            "concluiu_antes_do_corte": concluiu,
            "latencia_pensar_s": t_pensar,
            "latencia_serializar_s": t_serializar,
        })
        handoff_texto.append({
            "id": p["id"],
            "pergunta": p["pergunta"],
            "texto_parcial": texto_parcial,
            "concluiu_antes_do_corte": concluiu,
        })
        print(f"[A] problema {p['id']}: cache {nbytes / 1e6:.1f} MB, "
              f"concluiu_antes_do_corte={concluiu}", flush=True)

    with open(config.ARQ_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(config.ARQ_HANDOFF_TEXTO, "w", encoding="utf-8") as f:
        for h in handoff_texto:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"[A] {len(manifest)} handoffs gravados (N={n_corte})", flush=True)


if __name__ == "__main__":
    main()
