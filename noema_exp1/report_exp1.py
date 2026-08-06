"""Relatório do Experimento 1: tabela + curva bytes × acurácia."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noema_exp0"))
import config  # noqa: E402

DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"

ROTULOS = {
    "base_fp16": "cache completo fp16 (baseline = condição L do Exp 0)",
    "int8": "quantizado int8",
    "int4": "quantizado int4",
    "jan128": "janela: 4 âncoras + últimos 128 tokens",
    "jan64": "janela: 4 âncoras + últimos 64 tokens",
    "jan32": "janela: 4 âncoras + últimos 32 tokens",
    "cam24": "só as últimas 24 camadas (de 36)",
    "cam12": "só as últimas 12 camadas (de 36)",
    "int4_jan64": "int4 + janela de 64 (combinação agressiva)",
    "cam18": "só as últimas 18 camadas (de 36)",
    "int8_cam24": "int8 + últimas 24 camadas",
    "int4_cam24": "int4 + últimas 24 camadas",
}


def grafico(curva):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for p in curva:
        ax.scatter(p["bytes_medio"] / 1e6, p["acuracia"] * 100,
                   s=60, color="#4878b0", zorder=3)
        ax.annotate(p["tag"], (p["bytes_medio"] / 1e6, p["acuracia"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    base = next((p for p in curva if p["tag"] == "base_fp16"), None)
    if base:
        ax.axhline(base["acuracia"] * 100, ls="--", lw=1, color="#c44e52",
                   label=f"baseline {base['acuracia']:.0%}")
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Bytes transmitidos por handoff (MB, escala log)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Exp 1 — quanto pensamento sobrevive à compressão?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "curva.png", dpi=120)
    print(f"[report1] curva → {DIR_RESULTADOS / 'curva.png'}")


def relatorio(curva):
    base = next((p for p in curva if p["tag"] == "base_fp16"), None)
    arq_kl = DIR_RESULTADOS / "fidelidade_exp1.json"
    kls = json.load(open(arq_kl, encoding="utf-8")) if arq_kl.exists() else {}
    linhas = [
        "# Experimento 1 — Wire Format: relatório",
        "",
        f"Modelo: `{config.MODEL}` | 50 problemas GSM8K | caches do Exp 0 | "
        "geração greedy, seed 42",
        "",
        "| Configuração | Acurácia | MB/handoff | % do original | Handoff (s) | KL |",
        "|---|---|---|---|---|---|",
    ]
    original = base["bytes_medio"] if base else None
    for p in curva:
        pct = f"{p['bytes_medio'] / original * 100:.0f}%" if original else "—"
        kl = f"{kls[p['tag']]:.3f}" if p["tag"] in kls else "—"
        linhas.append(
            f"| {ROTULOS.get(p['tag'], p['tag'])} | {p['acuracia']:.1%} | "
            f"{p['bytes_medio'] / 1e6:.2f} | {pct} | {p['handoff_medio_s']:.3f} | {kl} |"
        )
    linhas += ["", "![curva](curva.png)", ""]

    saida = DIR_RESULTADOS / "relatorio_exp1.md"
    saida.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"[report1] relatório → {saida}")
    print("\n".join(linhas))


def main():
    curva = json.load(open(DIR_RESULTADOS / "curva.json", encoding="utf-8"))
    curva.sort(key=lambda p: -p["bytes_medio"])
    grafico(curva)
    relatorio(curva)


if __name__ == "__main__":
    main()
