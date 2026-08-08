"""Relatório do Experimento 4: custo da esteira latente vs. textual."""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent / "noema_exp0"))
sys.path.insert(0, str(RAIZ))

import config              # noqa: E402
from esteira import ETAPAS  # noqa: E402

DIR_RESULTADOS = RAIZ / "resultados"
NOMES = {"L": "L (cache entre agentes)", "T": "T (texto entre agentes)"}


def grafico(resumo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    papeis = [e["nome"] for e in ETAPAS]
    largura = 0.35
    for k, r in enumerate(resumo):
        xs = [i + (k - 0.5) * largura for i in range(len(papeis))]
        ax1.bar(xs, [e["prefill_s"] * 1000 for e in r["por_etapa"]],
                largura, label=NOMES[r["condicao"]])
        ax2.bar(xs, [e["tokens_recebidos"] for e in r["por_etapa"]],
                largura, label=NOMES[r["condicao"]])
    for ax, titulo, ylab in ((ax1, "Custo de 'entender o que chegou'", "prefill (ms)"),
                             (ax2, "Conteúdo retransmitido", "tokens recebidos")):
        ax.set_xticks(range(len(papeis)))
        ax.set_xticklabels(papeis)
        ax.set_ylabel(ylab)
        ax.set_title(titulo)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "esteira.png", dpi=120)
    print(f"[report4] gráfico → {DIR_RESULTADOS / 'esteira.png'}")


def main():
    resumo = json.load(open(DIR_RESULTADOS / "resumo.json", encoding="utf-8"))
    grafico(resumo)
    L = next(r for r in resumo if r["condicao"] == "L")
    T = next(r for r in resumo if r["condicao"] == "T")

    linhas = [
        "# Experimento 4 — A esteira: 3 agentes, um pensamento",
        "",
        f"Modelo: `{config.MODEL}` | {L['n']} problemas GSM8K | greedy, seed "
        f"{config.SEED} | esteira: " + " → ".join(e["nome"] for e in ETAPAS),
        "",
        "Os três agentes compartilham o checkpoint e diferem no papel. A porta de "
        "entrada (etapa 0) é idêntica nas duas vias; o que muda são os **dois "
        "saltos** seguintes.",
        "",
        "| Via | Acurácia | Conteúdo retransmitido | Cache trafegado | "
        "Prefill dos saltos | Handoff | Esteira completa |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in resumo:
        linhas.append(
            f"| {NOMES[r['condicao']]} | {r['acuracia']:.1%} | "
            f"{r['tokens_conteudo_por_esteira']:.0f} tokens | "
            f"{r['bytes_cache_por_esteira'] / 1e6:.2f} MB | "
            f"{r['prefill_saltos_s'] * 1000:.0f} ms | "
            f"{r['handoff_saltos_s'] * 1000:.0f} ms | "
            f"{r['esteira_total_s']:.2f} s |"
        )

    linhas += ["", "### Custo por etapa (prefill = reentender o que chegou)", "",
               "| Etapa | Papel | Prefill L | Prefill T | Tokens recebidos T |",
               "|---|---|---|---|---|"]
    for i, e in enumerate(ETAPAS):
        linhas.append(
            f"| {i} | {e['nome']} | {L['por_etapa'][i]['prefill_s'] * 1000:.0f} ms | "
            f"{T['por_etapa'][i]['prefill_s'] * 1000:.0f} ms | "
            f"{T['por_etapa'][i]['tokens_recebidos']:.0f} |"
        )

    custo_L = L["prefill_saltos_s"] + L["handoff_saltos_s"]
    custo_T = T["prefill_saltos_s"] + T["handoff_saltos_s"]
    ganho = (custo_T / custo_L) if custo_L > 0 else float("inf")
    linhas += [
        "", "![esteira](esteira.png)", "", "## Leitura", "",
        f"**Comunicação.** A esteira latente atravessou os dois saltos com "
        f"{L['tokens_conteudo_por_esteira']:.0f} tokens de conteúdo retransmitido "
        f"(só as instruções fixas de papel viajam como texto); a textual precisou "
        f"de {T['tokens_conteudo_por_esteira']:.0f} tokens em média — cada agente "
        f"relendo o problema e o trabalho dos anteriores.",
        "",
        f"**Custo de transferência.** Somando handoff + prefill nos dois saltos: "
        f"{custo_L * 1000:.0f} ms na via latente contra {custo_T * 1000:.0f} ms na "
        f"textual — **{ganho:.1f}× mais barato** passar o pensamento do que "
        f"reconstruí-lo. O gasto cresce com o comprimento da esteira: na via "
        f"textual o último agente relê tudo que veio antes, enquanto na latente o "
        f"custo de herdar é praticamente constante.",
        "",
        f"**Qualidade.** Acurácia final: {L['acuracia']:.1%} (latente) × "
        f"{T['acuracia']:.1%} (textual) — o barateamento "
        f"{'não custou qualidade' if L['acuracia'] >= T['acuracia'] else 'veio com perda de acurácia'}.",
        "",
    ]

    saida = DIR_RESULTADOS / "relatorio_exp4.md"
    saida.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"[report4] relatório → {saida}")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
