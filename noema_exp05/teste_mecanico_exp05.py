"""Smoke test mecânico do Exp 0.5, sem rede: pipeline A→h→B com modelo aleatório."""
import json
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

    env = dict(os.environ, NOEMA_MODEL=str(teste_mecanico.DIR_MODELO))
    subprocess.run([sys.executable, str(EXP0 / "agente_a.py"), "--n-corte", "16"],
                   check=True, cwd=EXP0, env=env)
    subprocess.run([sys.executable, str(RAIZ / "run_exp05.py")],
                   check=True, cwd=RAIZ, env=env)

    import metrics  # noqa: E402
    falhas = []
    for tag in ("h1", "h4"):
        linhas = metrics.ler_jsonl(RAIZ / "resultados" / f"resultados_{tag}.jsonl")
        if len(linhas) != 3:
            falhas.append(f"{tag}: {len(linhas)} linhas (esperava 3)")
        if any(l["bytes"] <= 0 for l in linhas):
            falhas.append(f"{tag}: bytes não registrados")
    est = json.load(open(RAIZ / "estados" / "manifest.json", encoding="utf-8"))
    if any(e["bytes"] > 100_000 for e in est):
        falhas.append("estados grandes demais — deveria ser KB, não MB")
    if falhas:
        for f in falhas:
            print("  -", f)
        raise SystemExit(1)
    print("\n[teste05] OK — pensamento contínuo validado mecanicamente")


if __name__ == "__main__":
    main()
