"""Experimento 0.5 — "Coconut caseiro": pensamento contínuo em vez de cache.

Em vez do KV-cache completo (MB), A transfere apenas os hidden states finais
dos últimos k tokens (~4 KB cada): o "pensamento vivo" na ponta do raciocínio.
B injeta esses vetores como inputs_embeds — pensamento contínuo — seguidos do
sufixo, e conclui. Máxima compressão, máxima perda esperada: o experimento
mede quanto sobrevive.

Condições: h1 (só o último token) e h4 (últimos 4). Referências: condição L do
Exp 0 (cache completo, teto) e Z (nada, piso).

Pré-requisito: Exp 0 rodado (problemas.jsonl + caches/manifest.json com n_corte).
Uso: python run_exp05.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent / "noema_exp0"))
import config      # noqa: E402
import metrics     # noqa: E402
import nucleo      # noqa: E402

DIR_RESULTADOS = RAIZ / "resultados"
DIR_ESTADOS = RAIZ / "estados"
K_ESTADOS = 4  # quantos hidden states finais A exporta


@torch.no_grad()
def agente_a05():
    """A pensa até o corte (mesmo N do Exp 0) e exporta os últimos K hidden states."""
    tok, model = nucleo.carregar_modelo()
    problemas = metrics.ler_jsonl(config.ARQ_PROBLEMAS)
    manifest0 = {m["id"]: m for m in
                 json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))}
    DIR_ESTADOS.mkdir(exist_ok=True)

    manifest = []
    for p in problemas:
        n_corte = manifest0[p["id"]]["n_corte"]
        msgs = [{"role": "user", "content": p["pergunta"] + "\nPense passo a passo."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(config.DEVICE)
        from transformers import DynamicCache
        cache = DynamicCache()
        ultimos = []

        def passo(x):
            out = model(x, past_key_values=cache, use_cache=True,
                        output_hidden_states=True,
                        attention_mask=torch.ones(
                            1, cache.get_seq_length() + x.shape[1],
                            dtype=torch.long, device=x.device))
            ultimos.append(out.hidden_states[-1][:, -1:].to(torch.float16).cpu())
            del ultimos[:-K_ESTADOS]
            return out.logits[:, -1].argmax(-1, keepdim=True)

        next_id = passo(ids)
        for _ in range(n_corte):
            if next_id.item() == tok.eos_token_id:
                break
            next_id = passo(next_id)

        h = torch.cat(ultimos, dim=1)  # [1, K, hidden]
        arq = DIR_ESTADOS / f"estado_{p['id']}.pt"
        torch.save({"h": h}, arq)
        manifest.append({"id": p["id"], "arquivo": arq.name,
                         "bytes": arq.stat().st_size})
        print(f"[A05] problema {p['id']}: {h.shape[1]} estados, "
              f"{arq.stat().st_size / 1e3:.1f} KB", flush=True)

    json.dump(manifest, open(DIR_ESTADOS / "manifest.json", "w",
                             encoding="utf-8"), indent=2)


@torch.no_grad()
def agente_b05(k: int):
    """B recebe só os k hidden states + sufixo, como pensamento contínuo."""
    tok, model = nucleo.carregar_modelo()
    manifest = json.load(open(DIR_ESTADOS / "manifest.json", encoding="utf-8"))
    emb = model.get_input_embeddings()
    ids_sufixo = tok(config.SUFIXO, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(config.DEVICE)

    linhas = []
    for m in manifest:
        h = torch.load(DIR_ESTADOS / m["arquivo"], map_location=config.DEVICE,
                       weights_only=True)["h"].to(config.DTYPE)[:, -k:, :]
        from transformers import DynamicCache
        cache = DynamicCache()
        x = torch.cat([h, emb(ids_sufixo)], dim=1)

        t0 = time.time()
        out = model(inputs_embeds=x, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones(1, x.shape[1], dtype=torch.long,
                                              device=config.DEVICE))
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        resposta = []
        for _ in range(config.MAX_NEW_B):
            if next_id.item() == tok.eos_token_id:
                break
            resposta.append(next_id)
            _, next_id = nucleo._passo(model, next_id, cache)
        texto = (tok.decode(torch.cat(resposta, dim=-1)[0],
                            skip_special_tokens=True) if resposta else "")

        linhas.append({"id": m["id"], "condicao": f"h{k}",
                       "resposta_gerada": texto,
                       "bytes": m["bytes"] * k // K_ESTADOS,
                       "latencia_total_s": time.time() - t0})
        print(f"[B05/h{k}] {m['id']}: {texto[:60]!r}", flush=True)
    return linhas


def corrigir_e_salvar(linhas, tag):
    gold = {p["id"]: p["gold"] for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS)}
    DIR_RESULTADOS.mkdir(exist_ok=True)
    with open(DIR_RESULTADOS / f"resultados_{tag}.jsonl", "w", encoding="utf-8") as f:
        for l in linhas:
            l["resposta_correta"] = gold[l["id"]]
            l["acertou"] = metrics.acertou(l["resposta_gerada"], gold[l["id"]])
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    acc = sum(l["acertou"] for l in linhas) / len(linhas)
    bts = sum(l["bytes"] for l in linhas) / len(linhas)
    print(f"[orq05] {tag}: acurácia {acc:.1%} | {bts / 1e3:.1f} KB médios")
    return {"tag": tag, "acuracia": acc, "bytes_medio": bts}


def main():
    if not config.ARQ_MANIFEST.exists():
        raise SystemExit("Rode o Exp 0 antes (precisa de problemas + manifest).")
    fase = sys.argv[1] if len(sys.argv) > 1 else None

    if fase == "--agente-a":
        agente_a05()
        return
    if fase in ("--h1", "--h4"):
        k = int(fase[3:])
        linhas = agente_b05(k)
        with open(DIR_RESULTADOS / f"brutos_h{k}.jsonl", "w", encoding="utf-8") as f:
            for l in linhas:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        return

    # orquestração: A e cada B em subprocessos separados (handoff via disco)
    DIR_RESULTADOS.mkdir(exist_ok=True)
    subprocess.run([sys.executable, __file__, "--agente-a"], check=True)
    resumo = []
    for k in (1, 4):
        subprocess.run([sys.executable, __file__, f"--h{k}"], check=True)
        linhas = metrics.ler_jsonl(DIR_RESULTADOS / f"brutos_h{k}.jsonl")
        resumo.append(corrigir_e_salvar(linhas, f"h{k}"))

    rel = ["# Experimento 0.5 — Pensamento contínuo (Coconut caseiro)", "",
           f"Modelo: `{config.MODEL}` | mesmos problemas e cortes do Exp 0", "",
           "| Condição | Acurácia | Bytes/handoff |", "|---|---|---|"]
    for r in resumo:
        rel.append(f"| {r['tag']} (últimos {r['tag'][1:]} hidden states) | "
                   f"{r['acuracia']:.1%} | {r['bytes_medio'] / 1e3:.1f} KB |")
    rel += ["", "Referências: condição L do Exp 0 (cache completo, ~9,4 MB) = 56%; "
            "condição Z (nada) = 0%.", ""]
    (DIR_RESULTADOS / "relatorio_exp05.md").write_text("\n".join(rel),
                                                      encoding="utf-8")
    print("\n".join(rel))


if __name__ == "__main__":
    main()
