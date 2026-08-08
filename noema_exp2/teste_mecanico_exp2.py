"""Smoke test mecânico do Exp 2, sem rede: dois modelos aleatórios de dimensões
DIFERENTES (64d e 48d) com o mesmo tokenizer, ponte ridge no meio."""
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
EXP0 = RAIZ.parent / "noema_exp0"
sys.path.insert(0, str(EXP0))

DIR_A = RAIZ / "modelo_teste_a"
DIR_B = RAIZ / "modelo_teste_b"


def main():
    import teste_mecanico  # noqa: E402
    teste_mecanico.construir_modelo_teste(destino=DIR_A, hidden=64, seed=42)
    teste_mecanico.construir_modelo_teste(destino=DIR_B, hidden=48, seed=43)
    teste_mecanico.preparar_problemas_fake()

    # textos de alinhamento locais (sem rede)
    textos = RAIZ / "textos_teste.jsonl"
    with open(textos, "w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({"texto": f"Maria tem {i} maçãs e ganha {i + 1}. "
                                          f"Total de {2 * i + 1} maçãs."}) + "\n")

    env = dict(os.environ, NOEMA_MODEL=str(DIR_A))
    subprocess.run([sys.executable, str(EXP0 / "agente_a.py"), "--n-corte", "16"],
                   check=True, cwd=EXP0, env=env)

    env2 = dict(os.environ, NOEMA_MODEL_A=str(DIR_A), NOEMA_MODEL_B=str(DIR_B),
                NOEMA_TEXTOS=str(textos), NOEMA_STRIDE="4")
    subprocess.run([sys.executable, str(RAIZ / "run_exp2.py")],
                   check=True, cwd=RAIZ, env=env2)

    import metrics  # noqa: E402
    falhas = []
    for cond in ("ponte", "controle", "teto"):
        linhas = metrics.ler_jsonl(RAIZ / "resultados" / f"resultados_{cond}.jsonl")
        if len(linhas) != 3:
            falhas.append(f"{cond}: {len(linhas)} linhas (esperava 3)")
    ad = __import__("torch").load(RAIZ / "dados" / "adaptador.pt",
                                  weights_only=True)
    if tuple(ad["W_ponte"].shape) != (65, 48):
        falhas.append(f"W_ponte com forma errada: {tuple(ad['W_ponte'].shape)}")
    if tuple(ad["W_self"].shape) != (49, 48):
        falhas.append(f"W_self com forma errada: {tuple(ad['W_self'].shape)}")
    if falhas:
        for f in falhas:
            print("  -", f)
        raise SystemExit(1)
    print(f"\n[teste2] OK — ponte 64d→48d validada mecanicamente "
          f"(W_ponte {tuple(ad['W_ponte'].shape)}, W_self {tuple(ad['W_self'].shape)})")


if __name__ == "__main__":
    main()
