"""Orquestrador do Experimento 4 — a esteira de 3 agentes.

Roda a mesma pipeline (extrator → calculador → verificador) por duas vias:
  L — os agentes passam o KV-cache entre si (entra token uma vez, sai uma vez)
  T — os agentes conversam por texto, cada um relendo tudo (o padrão de hoje)

Cada etapa é um processo separado; na via L o handoff passa por disco.

Uso:
  python run_exp4.py            # 3 etapas × 2 vias × 50 problemas
  python run_exp4.py --smoke    # 3 problemas
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent / "noema_exp0"))
sys.path.insert(0, str(RAIZ))

import config      # noqa: E402
import metrics     # noqa: E402
from esteira import ETAPAS  # noqa: E402

DIR_RESULTADOS = RAIZ / "resultados"


def rodar_etapa(i: int, cond: str, n: int):
    cmd = [sys.executable, str(RAIZ / "agente_etapa.py"),
           "--etapa", str(i), "--condicao", cond]
    print(f"[orq4] → {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=RAIZ,
                   env=dict(os.environ, NOEMA_EXP4_N=str(n)))


def resumir(cond: str):
    """Agrega a esteira inteira: acurácia do fim + custos somados nos 3 saltos."""
    gold = {p["id"]: p["gold"] for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS)}
    etapas = [metrics.ler_jsonl(DIR_RESULTADOS / f"etapa{i}_{cond}.jsonl")
              for i in range(len(ETAPAS))]
    n = len(etapas[0])

    final = etapas[-1]
    with open(DIR_RESULTADOS / f"resultados_{cond}.jsonl", "w", encoding="utf-8") as f:
        for k, l in enumerate(final):
            l["resposta_correta"] = gold[l["id"]]
            l["acertou"] = metrics.acertou(l["saida"], gold[l["id"]])
            l["latencia_esteira_s"] = sum(e[k]["latencia_etapa_s"] for e in etapas)
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    def soma(campo, etapas_alvo=None):
        alvo = etapas_alvo if etapas_alvo is not None else range(len(ETAPAS))
        return sum(l[campo] for i in alvo for l in etapas[i]) / n

    return {
        "condicao": cond,
        "n": n,
        "acuracia": sum(metrics.acertou(l["saida"], gold[l["id"]]) for l in final) / n,
        # saltos = etapas 1 e 2 (a 0 é a porta de entrada, igual nas duas vias)
        "tokens_conteudo_por_esteira": soma("tokens_conteudo_recebidos", [1, 2]),
        "bytes_cache_por_esteira": soma("bytes_cache"),
        "prefill_saltos_s": soma("latencia_prefill_s", [1, 2]),
        "handoff_saltos_s": soma("latencia_handoff_s", [1, 2]),
        "esteira_total_s": soma("latencia_etapa_s"),
        "por_etapa": [
            {"papel": ETAPAS[i]["nome"],
             "prefill_s": sum(l["latencia_prefill_s"] for l in etapas[i]) / n,
             "tokens_recebidos": sum(l["tokens_conteudo_recebidos"]
                                     for l in etapas[i]) / n}
            for i in range(len(ETAPAS))
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-problemas", type=int, default=None)
    args = ap.parse_args()

    n = args.n_problemas or (3 if args.smoke else config.N_PROBLEMAS)
    if not config.ARQ_PROBLEMAS.exists():
        raise SystemExit("Rode o Exp 0 antes (precisa de resultados/problemas.jsonl).")
    problemas = metrics.ler_jsonl(config.ARQ_PROBLEMAS)[:n]
    DIR_RESULTADOS.mkdir(exist_ok=True)
    if len(problemas) < n:
        print(f"[orq4] aviso: só há {len(problemas)} problemas preparados")

    resumo = []
    for cond in ("L", "T"):
        for i in range(len(ETAPAS)):
            rodar_etapa(i, cond, n)
        r = resumir(cond)
        resumo.append(r)
        print(f"[orq4] via {cond}: acurácia {r['acuracia']:.1%} | "
              f"{r['tokens_conteudo_por_esteira']:.0f} tokens de conteúdo | "
              f"prefill dos saltos {r['prefill_saltos_s']:.3f}s", flush=True)

    with open(DIR_RESULTADOS / "resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    subprocess.run([sys.executable, str(RAIZ / "report_exp4.py")], check=True,
                   cwd=RAIZ)


if __name__ == "__main__":
    main()
