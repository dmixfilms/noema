"""Processo B: conclui o raciocínio a partir do que recebeu de A.

Condições:
  L — herda o KV-cache do disco e recebe SÓ o sufixo "\\nResposta final:".
      Este modo nunca abre o arquivo de problemas: lê apenas manifest.json e os .pt.
  T — recebe o problema + raciocínio parcial como TEXTO (o canal textual clássico)
      e re-processa tudo do zero.
  Z — controle negativo: só o sufixo, sem cache e sem problema.

Saída: resultados/brutos_{condicao}.jsonl (sem gold — a correção é do orquestrador).
"""
import argparse
import json
import time

import torch

import config
import nucleo


def rodar_L():
    tok, _ = nucleo.carregar_modelo()
    manifest = json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))
    ids_sufixo = tok(config.SUFIXO, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(config.DEVICE)
    linhas = []
    for m in manifest:
        t0 = time.time()
        cache = nucleo.carregar_cache(str(config.DIR_CACHES / m["cache_file"]))
        t_handoff = time.time() - t0

        t0 = time.time()
        resposta, _ = nucleo.gerar_greedy(ids_sufixo, cache=cache)
        t_gerar = time.time() - t0

        linhas.append({
            "id": m["id"],
            "condicao": "L",
            "resposta_gerada": resposta,
            "tokens_texto_A_para_B": 0,
            "bytes_cache": m["bytes_cache"],
            "latencia_handoff_s": t_handoff,
            "latencia_total_s": t_handoff + t_gerar,
            "concluiu_antes_do_corte": m["concluiu_antes_do_corte"],
        })
        print(f"[B/L] {m['id']}: {resposta[:70]!r}", flush=True)
    return linhas


def rodar_T():
    tok, _ = nucleo.carregar_modelo()
    handoffs = [json.loads(l) for l in open(config.ARQ_HANDOFF_TEXTO, encoding="utf-8")]
    linhas = []
    for h in handoffs:
        # B reconstrói o contexto a partir do texto recebido: mesmo template de A
        # + raciocínio parcial re-tokenizado + sufixo. A re-tokenização é a
        # serialização com perda que a via textual impõe.
        msgs = [{"role": "user", "content": h["pergunta"] + "\nPense passo a passo."}]
        ids_prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             return_tensors="pt")
        ids_cont = tok(h["texto_parcial"] + config.SUFIXO, return_tensors="pt",
                       add_special_tokens=False).input_ids
        ids = torch.cat([ids_prompt, ids_cont], dim=-1).to(config.DEVICE)

        # Custo do canal: a mensagem que A precisa mandar p/ B (problema + raciocínio).
        tokens_a_para_b = len(
            tok(h["pergunta"] + h["texto_parcial"], add_special_tokens=False).input_ids
        )

        t0 = time.time()
        resposta, t_prefill = nucleo.gerar_greedy(ids)
        t_total = time.time() - t0

        linhas.append({
            "id": h["id"],
            "condicao": "T",
            "resposta_gerada": resposta,
            "tokens_texto_A_para_B": tokens_a_para_b,
            "bytes_cache": 0,
            "latencia_handoff_s": t_prefill,  # custo de B re-processar o contexto
            "latencia_total_s": t_total,
            "concluiu_antes_do_corte": h["concluiu_antes_do_corte"],
        })
        print(f"[B/T] {h['id']}: {resposta[:70]!r}", flush=True)
    return linhas


def rodar_Z():
    tok, _ = nucleo.carregar_modelo()
    # Mesmo universo de ids das outras condições, mas sem cache e sem problema.
    manifest = json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))
    ids = tok("Resposta final:", return_tensors="pt",
              add_special_tokens=False).input_ids.to(config.DEVICE)
    linhas = []
    for m in manifest:
        t0 = time.time()
        resposta, _ = nucleo.gerar_greedy(ids)
        t_total = time.time() - t0
        linhas.append({
            "id": m["id"],
            "condicao": "Z",
            "resposta_gerada": resposta,
            "tokens_texto_A_para_B": 0,
            "bytes_cache": 0,
            "latencia_handoff_s": 0.0,
            "latencia_total_s": t_total,
            "concluiu_antes_do_corte": m["concluiu_antes_do_corte"],
        })
        print(f"[B/Z] {m['id']}: {resposta[:70]!r}", flush=True)
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condicao", choices=["T", "L", "Z"], required=True)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    linhas = {"L": rodar_L, "T": rodar_T, "Z": rodar_Z}[args.condicao]()

    saida = config.DIR_RESULTADOS / f"brutos_{args.condicao}.jsonl"
    with open(saida, "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"[B/{args.condicao}] {len(linhas)} respostas gravadas em {saida}", flush=True)


if __name__ == "__main__":
    main()
