"""Smoke test mecânico do Exp 4, sem rede: esteira de 3 etapas × 2 vias com o
modelo aleatório minúsculo."""
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
EXP0 = RAIZ.parent / "noema_exp0"
sys.path.insert(0, str(EXP0))


def main():
    import teste_mecanico  # noqa: E402
    teste_mecanico.construir_modelo_teste()
    teste_mecanico.preparar_problemas_fake()

    env = dict(os.environ, NOEMA_MODEL=str(teste_mecanico.DIR_MODELO),
               NOEMA_MAX_NEW_B="24")
    subprocess.run([sys.executable, str(RAIZ / "run_exp4.py"), "--smoke"],
                   check=True, cwd=RAIZ, env=env)

    import json
    import metrics  # noqa: E402
    falhas = []
    resumo = json.load(open(RAIZ / "resultados" / "resumo.json", encoding="utf-8"))
    L = next(r for r in resumo if r["condicao"] == "L")
    T = next(r for r in resumo if r["condicao"] == "T")
    for cond in ("L", "T"):
        linhas = metrics.ler_jsonl(RAIZ / "resultados" / f"resultados_{cond}.jsonl")
        if len(linhas) != 3:
            falhas.append(f"{cond}: {len(linhas)} linhas (esperava 3)")
    if L["tokens_conteudo_por_esteira"] != 0:
        falhas.append("via L retransmitiu conteúdo (deveria ser 0)")
    if T["tokens_conteudo_por_esteira"] <= 0:
        falhas.append("via T sem tokens de conteúdo registrados")
    if L["bytes_cache_por_esteira"] <= 0:
        falhas.append("via L sem bytes de cache registrados")
    if not (RAIZ / "resultados" / "relatorio_exp4.md").exists():
        falhas.append("relatório não gerado")
    if falhas:
        for f in falhas:
            print("  -", f)
        raise SystemExit(1)
    print("\n[teste4] OK — esteira validada: L com 0 tokens de conteúdo entre "
          f"agentes ({L['bytes_cache_por_esteira'] / 1e3:.0f} KB de cache), "
          f"T com {T['tokens_conteudo_por_esteira']:.0f} tokens")


if __name__ == "__main__":
    main()
