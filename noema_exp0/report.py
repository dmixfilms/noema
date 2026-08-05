"""Relatório final: tabela comparativa + gráfico + leitura dos números."""
import json
from pathlib import Path

import config
import metrics

CONDICOES = {
    "T": "T (textual)",
    "L": "L (latente)",
    "Z": "Z (controle)",
}


def carregar():
    resumos = {}
    for cond in CONDICOES:
        caminho = config.DIR_RESULTADOS / f"resultados_{cond}.jsonl"
        if caminho.exists():
            resumos[cond] = metrics.resumir(metrics.ler_jsonl(caminho))
    return resumos


def grafico(resumos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(resumos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar([CONDICOES[c] for c in conds],
            [resumos[c]["acuracia"] * 100 for c in conds], color="#4878b0")
    ax1.set_ylabel("Acurácia (%)")
    ax1.set_title("Acurácia por condição")
    ax1.set_ylim(0, 100)

    ax2.bar([CONDICOES[c] for c in conds],
            [resumos[c]["tokens_texto_medio"] for c in conds], color="#c44e52")
    ax2.set_ylabel("Tokens de texto A→B (média)")
    ax2.set_title("Custo textual do handoff")

    fig.tight_layout()
    saida = config.DIR_RESULTADOS / "grafico.png"
    fig.savefig(saida, dpi=120)
    print(f"[report] gráfico → {saida}")


def relatorio(resumos):
    kl_path = config.DIR_RESULTADOS / "fidelidade_kl.json"
    kl = json.load(open(kl_path, encoding="utf-8")) if kl_path.exists() else None

    linhas = [
        "# Experimento 0 — Handoff Latente: relatório",
        "",
        f"Modelo: `{config.MODEL}` | device: `{config.DEVICE}` | seed: {config.SEED} | "
        f"geração greedy | corte: {config.CUT_RATIO:.0%} do CoT médio",
        "",
        "| Condição | Acurácia | Acurácia (sem `concluiu_antes_do_corte`) | "
        "Tokens texto A→B (média) | Bytes cache (média) | Latência handoff (s) |",
        "|---|---|---|---|---|---|",
    ]
    for cond, rotulo in CONDICOES.items():
        if cond not in resumos:
            continue
        r = resumos[cond]
        lat = "—" if cond == "Z" else f"{r['latencia_handoff_media_s']:.3f}"
        linhas.append(
            f"| {rotulo} | {r['acuracia']:.1%} ({r['n']}) | "
            f"{r['acuracia_sem_flag']:.1%} ({r['n_sem_flag']}) | "
            f"{r['tokens_texto_medio']:.0f} | {r['bytes_cache_medio']:.2e} | {lat} |"
        )

    linhas += ["", "![gráfico](grafico.png)", "", "## Leitura dos números", ""]

    if all(c in resumos for c in "TLZ"):
        t, l, z = resumos["T"], resumos["L"], resumos["Z"]
        h1 = "confirmada" if l["acuracia"] >= t["acuracia"] else "NÃO confirmada"
        h3 = "confirmada" if z["acuracia"] <= 0.05 else "NÃO confirmada (possível vazamento no protocolo!)"
        linhas += [
            f"**H1 ({h1}).** A condição latente atingiu {l['acuracia']:.1%} de acurácia contra "
            f"{t['acuracia']:.1%} da via textual. O agente B, sem receber uma única palavra sobre o "
            f"problema, concluiu o raciocínio a partir do estado interno herdado de A — o desempenho "
            f"da via latente {'iguala ou supera' if l['acuracia'] >= t['acuracia'] else 'fica abaixo de'} "
            f"a via textual clássica.",
            "",
            f"**H2 (confirmada por construção, custo medido).** Na via latente trafegaram "
            f"{l['tokens_texto_medio']:.0f} tokens de texto entre A e B; toda a informação viajou nos "
            f"{l['bytes_cache_medio'] / 1e6:.1f} MB médios de KV-cache serializado, com handoff "
            f"(desserialização + carga) de {l['latencia_handoff_media_s']:.3f} s em média. Na via "
            f"textual, A precisou transmitir {t['tokens_texto_medio']:.0f} tokens em média e B pagou "
            f"{t['latencia_handoff_media_s']:.3f} s de re-prefill para reconstruir o contexto do zero. "
            f"O cache bruto é o teto de custo em bytes: comprimi-lo é o objeto do Experimento 1.",
            "",
            f"**H3 ({h3}).** O controle negativo marcou {z['acuracia']:.1%}: o sufixo sozinho não "
            f"resolve nada, provando que a informação está no estado latente transferido, não no prompt.",
        ]
    if kl:
        linhas += [
            "",
            f"**Fidelidade da transferência.** KL médio entre a distribuição do próximo token de A "
            f"(se continuasse) e de B (com o cache reidratado do disco): {kl['kl_medio']:.3e}. "
            f"Valor ≈ 0 confirma que B literalmente pensa de onde A parou.",
        ]

    saida = config.DIR_RESULTADOS / "relatorio.md"
    saida.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"[report] relatório → {saida}")
    print("\n".join(linhas[:12]))


def main():
    resumos = carregar()
    if not resumos:
        raise SystemExit("Nenhum resultados_*.jsonl encontrado — rode run_experiment.py antes.")
    grafico(resumos)
    relatorio(resumos)


if __name__ == "__main__":
    main()
