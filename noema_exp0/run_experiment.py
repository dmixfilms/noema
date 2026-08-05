"""Orquestrador do Experimento 0.

Fluxo:
  1. Prepara os primeiros N problemas do GSM8K (split test) em problemas.jsonl.
  2. Roda o agente A em SUBPROCESSO (piloto de calibração de N + handoffs T/L).
  3. Roda o agente B em SUBPROCESSO, uma vez por condição (L, T, Z).
     A separação em processos é obrigatória: o handoff latente passa por disco.
  4. Corrige as respostas (join com o gold) e grava resultados_{cond}.jsonl.
  5. (opcional, --kl) Fidelidade da transferência: KL entre a distribuição do
     próximo token de A-se-continuasse vs. B-com-cache-herdado-do-disco.
  6. Gera o relatório (report.py).

Uso:
  python run_experiment.py --smoke      # 3 problemas, N fixo=80 (valida o protocolo)
  python run_experiment.py              # rodada completa: 50 × 3 condições
  python run_experiment.py --kl         # + métrica bônus de fidelidade
"""
import argparse
import json
import random
import subprocess
import sys

import torch

import config
import metrics


def preparar_problemas(n: int):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(config.ARQ_PROBLEMAS, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = ds[i]
            f.write(json.dumps({
                "id": i,
                "pergunta": ex["question"],
                "gold": metrics.extrair_gold(ex["answer"]),
            }, ensure_ascii=False) + "\n")
    print(f"[orq] {n} problemas gravados em {config.ARQ_PROBLEMAS}")


def rodar_subprocesso(script: str, *args: str):
    cmd = [sys.executable, str(config.RAIZ / script), *args]
    print(f"[orq] → {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=config.RAIZ)


def corrigir(condicao: str):
    gold = {p["id"]: p["gold"] for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS)}
    brutos = metrics.ler_jsonl(config.DIR_RESULTADOS / f"brutos_{condicao}.jsonl")
    saida = config.DIR_RESULTADOS / f"resultados_{condicao}.jsonl"
    with open(saida, "w", encoding="utf-8") as f:
        for l in brutos:
            l["resposta_correta"] = gold[l["id"]]
            l["acertou"] = metrics.acertou(l["resposta_gerada"], gold[l["id"]])
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    acc = metrics.resumir(metrics.ler_jsonl(saida))["acuracia"]
    print(f"[orq] condição {condicao}: acurácia {acc:.2%} → {saida}")


@torch.no_grad()
def fidelidade_kl(n_problemas: int = 10):
    """B literalmente pensa de onde A parou? Para cada problema: A avança um passo
    a partir do corte; B (com o cache reidratado do DISCO) dá o mesmo passo.
    KL(P_A || P_B) ≈ 0 confirma a fidelidade do handoff."""
    import torch.nn.functional as F
    import nucleo

    tok, model = nucleo.carregar_modelo()
    problemas = metrics.ler_jsonl(config.ARQ_PROBLEMAS)[:n_problemas]
    manifest = {m["id"]: m for m in json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))}

    kls = []
    for p in problemas:
        n_corte = manifest[p["id"]]["n_corte"]
        cache_a, _, _, pendente = nucleo.agente_A_pensa(p["pergunta"], n_corte)

        caminho = config.DIR_CACHES / f"kl_tmp_{p['id']}.pt"
        nucleo.serializar_cache(cache_a, str(caminho))
        cache_b = nucleo.carregar_cache(str(caminho))
        caminho.unlink()

        logits_a, _ = nucleo._passo(model, pendente, cache_a)
        logits_b, _ = nucleo._passo(model, pendente, cache_b)
        log_pa = F.log_softmax(logits_a.float(), dim=-1)
        log_pb = F.log_softmax(logits_b.float(), dim=-1)
        kl = F.kl_div(log_pb, log_pa, log_target=True, reduction="sum").item()
        kls.append({"id": p["id"], "kl": kl})
        print(f"[orq/KL] problema {p['id']}: KL = {kl:.3e}", flush=True)

    media = sum(k["kl"] for k in kls) / len(kls)
    saida = config.DIR_RESULTADOS / "fidelidade_kl.json"
    json.dump({"kl_medio": media, "por_problema": kls},
              open(saida, "w", encoding="utf-8"), indent=2)
    print(f"[orq/KL] KL médio = {media:.3e} → {saida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="3 problemas com N fixo=80 (sem piloto)")
    ap.add_argument("--n-problemas", type=int, default=None)
    ap.add_argument("--n-corte", type=int, default=None,
                    help="pula o piloto e usa este N fixo")
    ap.add_argument("--kl", action="store_true",
                    help="métrica bônus: fidelidade da transferência (10 problemas)")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)

    n = args.n_problemas or (3 if args.smoke else config.N_PROBLEMAS)
    n_corte = args.n_corte or (80 if args.smoke else None)

    preparar_problemas(n)

    args_a = ["--n-corte", str(n_corte)] if n_corte else []
    rodar_subprocesso("agente_a.py", *args_a)

    for cond in ["L", "T", "Z"]:
        rodar_subprocesso("agente_b.py", "--condicao", cond)
        corrigir(cond)

    if args.kl:
        fidelidade_kl(min(10, n))

    rodar_subprocesso("report.py")


if __name__ == "__main__":
    main()
