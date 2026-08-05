"""Orquestrador do Experimento 1 — curvas de degradação do wire format.

Pré-requisito: os caches do Exp 0 em noema_exp0/caches/ (agente A já rodado).
Para cada configuração de compressão, roda o agente B1 em subprocesso, corrige
contra o gold e agrega a curva bytes × acurácia.

Uso:
  python run_exp1.py            # todas as configurações
  python run_exp1.py --configs base_fp16,int8,int4
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noema_exp0"))
import config      # noqa: E402
import metrics     # noqa: E402

DIR_EXP1 = Path(__file__).resolve().parent
DIR_RESULTADOS = DIR_EXP1 / "resultados"

# Eixos independentes + uma combinação agressiva. base_fp16 reproduz a condição
# L do Exp 0 (sanidade: acurácia deve bater com o baseline de lá).
CONFIGS = [
    {"tag": "base_fp16"},
    {"tag": "int8", "quant": "int8"},
    {"tag": "int4", "quant": "int4"},
    {"tag": "jan128", "janela": 128},
    {"tag": "jan64", "janela": 64},
    {"tag": "jan32", "janela": 32},
    {"tag": "cam24", "camadas": 24},
    {"tag": "cam12", "camadas": 12},
    {"tag": "int4_jan64", "quant": "int4", "janela": 64},
    # Refino pós-primeira rodada: empilhar os eixos vencedores (quantização ×
    # camadas) e localizar o penhasco entre 12 e 24 camadas.
    {"tag": "cam18", "camadas": 18},
    {"tag": "int8_cam24", "quant": "int8", "camadas": 24},
    {"tag": "int4_cam24", "quant": "int4", "camadas": 24},
]

# Configurações do teste mecânico (modelo minúsculo) — só rodam se pedidas
# explicitamente via --configs, nunca na rodada padrão.
CONFIGS_TESTE = [
    {"tag": "jan8_teste", "janela": 8},
    {"tag": "cam1_teste", "camadas": 1},
]


def rodar_config(cfg: dict):
    cmd = [sys.executable, str(DIR_EXP1 / "agente_b1.py"), "--tag", cfg["tag"]]
    if cfg.get("quant"):
        cmd += ["--quant", cfg["quant"]]
    if cfg.get("janela"):
        cmd += ["--janela", str(cfg["janela"])]
    if cfg.get("camadas"):
        cmd += ["--camadas", str(cfg["camadas"])]
    print(f"[orq1] → {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=DIR_EXP1)


def corrigir(tag: str) -> dict:
    gold = {p["id"]: p["gold"] for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS)}
    brutos = metrics.ler_jsonl(DIR_RESULTADOS / f"brutos_{tag}.jsonl")
    saida = DIR_RESULTADOS / f"resultados_{tag}.jsonl"
    with open(saida, "w", encoding="utf-8") as f:
        for l in brutos:
            l["resposta_correta"] = gold[l["id"]]
            l["acertou"] = metrics.acertou(l["resposta_gerada"], gold[l["id"]])
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    n = len(brutos)
    acc = sum(l["acertou"] for l in brutos) / n if n else 0.0
    bytes_medio = sum(l["bytes_wire"] for l in brutos) / n if n else 0.0
    resumo = {"tag": tag, "n": n, "acuracia": acc, "bytes_medio": bytes_medio,
              "handoff_medio_s": sum(l["latencia_handoff_s"] for l in brutos) / n}
    print(f"[orq1] {tag}: acurácia {acc:.1%} | {bytes_medio / 1e6:.2f} MB médios")
    return resumo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None,
                    help="lista de tags separadas por vírgula (default: todas)")
    args = ap.parse_args()

    if not config.ARQ_MANIFEST.exists():
        raise SystemExit(
            "Caches do Exp 0 não encontrados em noema_exp0/caches/. "
            "Rode antes: cd ../noema_exp0 && python agente_a.py"
        )

    tags = set(args.configs.split(",")) if args.configs else None
    universo = CONFIGS + (CONFIGS_TESTE if tags else [])
    configs = [c for c in universo if tags is None or c["tag"] in tags]

    # Rodadas parciais (--configs) acumulam sobre a curva existente em vez de
    # sobrescrevê-la; pontos do teste mecânico (_teste) ficam de fora.
    arq_curva = DIR_RESULTADOS / "curva.json"
    DIR_RESULTADOS.mkdir(exist_ok=True)
    pontos = ({p["tag"]: p for p in json.load(open(arq_curva, encoding="utf-8"))}
              if arq_curva.exists() else {})
    for cfg in configs:
        rodar_config(cfg)
        pontos[cfg["tag"]] = corrigir(cfg["tag"])

    curva = [p for t, p in pontos.items() if not t.endswith("_teste")]
    with open(arq_curva, "w", encoding="utf-8") as f:
        json.dump(curva, f, ensure_ascii=False, indent=2)
    print(f"[orq1] curva com {len(curva)} pontos → {DIR_RESULTADOS / 'curva.json'}")

    subprocess.run([sys.executable, str(DIR_EXP1 / "report_exp1.py")],
                   check=True, cwd=DIR_EXP1)


if __name__ == "__main__":
    main()
