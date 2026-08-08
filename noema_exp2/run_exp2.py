"""Experimento 2 — Interlíngua: handoff entre modelos DIFERENTES.

O agente A (Qwen2.5-3B) pensa; o agente B (Qwen2.5-1.5B — outro modelo, outras
dimensões) conclui. A ponte é um adaptador linear W treinado por regressão
ridge para traduzir hidden states do espaço de A (2048d) para o de B (1536d),
usando textos do split TRAIN do GSM8K (nunca o test). O canal é o mesmo do
Exp 0.5: os últimos K hidden states viajam como pensamento contínuo.

Condições na avaliação (50 problemas do test):
  ponte     — h_A × W → B                (a tese)
  controle  — h_A × W_aleatória → B      (a ponte importa, ou qualquer vetor serve?)
  teto      — B pensa sozinho e usa os próprios h  (limite superior deste canal)

Fases em subprocessos separados (1 modelo por vez na VRAM):
  python run_exp2.py            # tudo: coleta A/B → treina W → avalia 3 condições
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent
EXP0 = RAIZ.parent / "noema_exp0"
sys.path.insert(0, str(EXP0))

MODEL_A = os.environ.get("NOEMA_MODEL_A", "Qwen/Qwen2.5-3B-Instruct")
MODEL_B = os.environ.get("NOEMA_MODEL_B", "Qwen/Qwen2.5-1.5B-Instruct")

DIR_DADOS = RAIZ / "dados"
DIR_RESULTADOS = RAIZ / "resultados"
N_TEXTOS_TREINO = 400
MAX_TOK_TREINO = 192
STRIDE = int(os.environ.get("NOEMA_STRIDE", "16"))  # 1 estado a cada STRIDE tokens
K_ESTADOS = 4
RIDGE_LAMBDA = 1.0
SEED = 42


def _textos_treino():
    """Textos de alinhamento: GSM8K train (ou arquivo local via NOEMA_TEXTOS)."""
    local = os.environ.get("NOEMA_TEXTOS")
    if local:
        import metrics
        return [t["texto"] for t in metrics.ler_jsonl(local)][:N_TEXTOS_TREINO]
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    return [ds[i]["question"] + "\n" + ds[i]["answer"]
            for i in range(N_TEXTOS_TREINO)]


@torch.no_grad()
def coletar(lado: str):
    """Passa os textos de treino pelo modelo e colhe, em posições fixas (as
    MESMAS nos dois lados — mesmo tokenizer): o hidden state final na posição t
    e o embedding de entrada do token t+1. O Exp 0.5 mostrou que hidden states
    injetados crus degeneram (fora de distribuição); por isso o adaptador é
    treinado para decodificar estado → embedding do próximo token ("palavra
    suave"), que é entrada nativa do receptor."""
    import config
    import nucleo
    tok, model = nucleo.carregar_modelo()
    emb = model.get_input_embeddings()

    estados, alvos_emb, posicoes_usadas = [], [], []
    for i, texto in enumerate(_textos_treino()):
        ids = tok(texto, return_tensors="pt", truncation=True,
                  max_length=MAX_TOK_TREINO).input_ids.to(config.DEVICE)
        if ids.shape[1] < STRIDE + 1:
            continue
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0]  # [seq, hidden]
        pos = list(range(STRIDE - 1, ids.shape[1] - 1, STRIDE))
        estados.append(h[pos].to(torch.float32).cpu())
        alvos_emb.append(emb(ids[0, [p + 1 for p in pos]]).to(torch.float32).cpu())
        posicoes_usadas.append(pos)
        if i % 100 == 0:
            print(f"[coleta/{lado}] {i}/{N_TEXTOS_TREINO}", flush=True)

    DIR_DADOS.mkdir(exist_ok=True)
    torch.save({"estados": torch.cat(estados), "alvos_emb": torch.cat(alvos_emb),
                "posicoes": posicoes_usadas},
               DIR_DADOS / f"estados_{lado}.pt")
    print(f"[coleta/{lado}] {sum(len(p) for p in posicoes_usadas)} estados "
          f"de {N_TEXTOS_TREINO} textos", flush=True)


def _ridge(X, Y):
    Xa = torch.cat([X, torch.ones(len(X), 1)], dim=1)
    XtX = Xa.T @ Xa + RIDGE_LAMBDA * torch.eye(Xa.shape[1])
    W = torch.linalg.solve(XtX, Xa.T @ Y)
    residuo = ((Xa @ W - Y).pow(2).mean().sqrt() / Y.pow(2).mean().sqrt()).item()
    return W, residuo


def treinar():
    """Dois adaptadores ridge closed-form, ambos decodificando para o espaço de
    EMBEDDINGS de B (entrada nativa do receptor):
      W_ponte: h_A → emb_B(próximo token)   — a interlíngua
      W_self:  h_B → emb_B(próximo token)   — o teto justo (mesmo canal, sem
                                              travessia entre modelos)"""
    torch.manual_seed(SEED)
    a = torch.load(DIR_DADOS / "estados_A.pt", weights_only=True)
    b = torch.load(DIR_DADOS / "estados_B.pt", weights_only=True)
    assert a["posicoes"] == b["posicoes"], "posições de coleta divergem entre A e B"
    W_ponte, r1 = _ridge(a["estados"], b["alvos_emb"])
    W_self, r2 = _ridge(b["estados"], b["alvos_emb"])
    torch.save({"W_ponte": W_ponte, "W_self": W_self}, DIR_DADOS / "adaptador.pt")
    print(f"[treino] W_ponte {tuple(W_ponte.shape)} (erro rel. {r1:.3f}) | "
          f"W_self {tuple(W_self.shape)} (erro rel. {r2:.3f})")


@torch.no_grad()
def avaliar_a():
    """A pensa nos 50 problemas do test (mesmo n_corte do Exp 0) e exporta h_A."""
    import config
    import metrics
    import nucleo
    tok, model = nucleo.carregar_modelo()
    problemas = metrics.ler_jsonl(config.ARQ_PROBLEMAS)
    manifest0 = {m["id"]: m for m in
                 json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))}

    from transformers import DynamicCache
    saida = {}
    for p in problemas:
        msgs = [{"role": "user", "content": p["pergunta"] + "\nPense passo a passo."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(config.DEVICE)
        cache, ultimos = DynamicCache(), []

        def passo(x):
            out = model(x, past_key_values=cache, use_cache=True,
                        output_hidden_states=True,
                        attention_mask=torch.ones(
                            1, cache.get_seq_length() + x.shape[1],
                            dtype=torch.long, device=x.device))
            ultimos.append(out.hidden_states[-1][:, -1:].to(torch.float32).cpu())
            del ultimos[:-K_ESTADOS]
            return out.logits[:, -1].argmax(-1, keepdim=True)

        next_id = passo(ids)
        for _ in range(manifest0[p["id"]]["n_corte"]):
            if next_id.item() == tok.eos_token_id:
                break
            next_id = passo(next_id)
        saida[p["id"]] = torch.cat(ultimos, dim=1)
        print(f"[aval/A] problema {p['id']} pensado", flush=True)

    DIR_DADOS.mkdir(exist_ok=True)
    torch.save(saida, DIR_DADOS / "eval_estados_A.pt")


@torch.no_grad()
def avaliar_b(cond: str):
    """B (o modelo receptor) conclui a partir do pensamento traduzido."""
    import config
    import metrics
    import nucleo
    tok, model = nucleo.carregar_modelo()
    problemas = metrics.ler_jsonl(config.ARQ_PROBLEMAS)
    manifest0 = {m["id"]: m for m in
                 json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))}
    emb = model.get_input_embeddings()
    dim_b = model.config.hidden_size
    ids_sufixo = tok(config.SUFIXO, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(config.DEVICE)

    adaptadores = torch.load(DIR_DADOS / "adaptador.pt", weights_only=True)
    if cond in ("ponte", "controle"):
        estados_a = torch.load(DIR_DADOS / "eval_estados_A.pt", weights_only=True)
        W = adaptadores["W_ponte"]
        if cond == "controle":
            torch.manual_seed(SEED)
            W = torch.randn_like(W) * W.std()
    else:
        W_self = adaptadores["W_self"]

    from transformers import DynamicCache
    linhas = []
    for p in problemas:
        if cond == "teto":
            # B pensa sozinho até o mesmo corte e usa os próprios h (Exp 0.5 em B)
            msgs = [{"role": "user",
                     "content": p["pergunta"] + "\nPense passo a passo."}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt").to(config.DEVICE)
            cache_t, ultimos = DynamicCache(), []

            def passo(x):
                out = model(x, past_key_values=cache_t, use_cache=True,
                            output_hidden_states=True,
                            attention_mask=torch.ones(
                                1, cache_t.get_seq_length() + x.shape[1],
                                dtype=torch.long, device=x.device))
                ultimos.append(out.hidden_states[-1][:, -1:])
                del ultimos[:-K_ESTADOS]
                return out.logits[:, -1].argmax(-1, keepdim=True)

            nid = passo(ids)
            for _ in range(manifest0[p["id"]]["n_corte"]):
                if nid.item() == tok.eos_token_id:
                    break
                nid = passo(nid)
            h_b = torch.cat(ultimos, dim=1)[0].to(torch.float32).cpu()  # [K, dim_b]
            X = torch.cat([h_b, torch.ones(h_b.shape[0], 1)], dim=1)
            h = (X @ W_self).unsqueeze(0).to(config.DEVICE, config.DTYPE)
        else:
            h_a = estados_a[p["id"]][0]  # [K, dim_a]
            X = torch.cat([h_a, torch.ones(h_a.shape[0], 1)], dim=1)
            h = (X @ W).unsqueeze(0).to(config.DEVICE, config.DTYPE)

        cache = DynamicCache()
        x = torch.cat([h.to(config.DEVICE), emb(ids_sufixo)], dim=1)
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
        linhas.append({"id": p["id"], "condicao": cond, "resposta_gerada": texto,
                       "bytes": h.numel() * 2, "latencia_total_s": time.time() - t0})
        print(f"[aval/B/{cond}] {p['id']}: {texto[:55]!r}", flush=True)

    DIR_RESULTADOS.mkdir(exist_ok=True)
    with open(DIR_RESULTADOS / f"brutos_{cond}.jsonl", "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


def _sub(fase, modelo, extra=()):
    env = dict(os.environ, NOEMA_MODEL=modelo)
    subprocess.run([sys.executable, __file__, fase, *extra], check=True, env=env)


def main():
    import config
    import metrics
    if len(sys.argv) > 1:
        fase = sys.argv[1]
        if fase == "--coletar":
            coletar(sys.argv[2])
        elif fase == "--treinar":
            treinar()
        elif fase == "--avaliar-a":
            avaliar_a()
        elif fase == "--avaliar-b":
            avaliar_b(sys.argv[2])
        return

    if not config.ARQ_MANIFEST.exists():
        raise SystemExit("Rode o Exp 0 antes (precisa de problemas + manifest).")

    print(f"[orq2] A = {MODEL_A}\n[orq2] B = {MODEL_B}", flush=True)
    _sub("--coletar", MODEL_A, ("A",))
    _sub("--coletar", MODEL_B, ("B",))
    _sub("--treinar", MODEL_B)
    _sub("--avaliar-a", MODEL_A)
    for cond in ("ponte", "controle", "teto"):
        _sub("--avaliar-b", MODEL_B, (cond,))

    gold = {p["id"]: p["gold"] for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS)}
    rel = ["# Experimento 2 — Interlíngua: relatório", "",
           f"A (pensa): `{MODEL_A}` → B (conclui): `{MODEL_B}` | adaptador ridge "
           f"treinado com {N_TEXTOS_TREINO} textos do GSM8K train", "",
           "| Condição | Acurácia | Bytes/handoff |", "|---|---|---|"]
    for cond in ("ponte", "controle", "teto"):
        linhas = metrics.ler_jsonl(DIR_RESULTADOS / f"brutos_{cond}.jsonl")
        saida = DIR_RESULTADOS / f"resultados_{cond}.jsonl"
        with open(saida, "w", encoding="utf-8") as f:
            for l in linhas:
                l["resposta_correta"] = gold[l["id"]]
                l["acertou"] = metrics.acertou(l["resposta_gerada"], gold[l["id"]])
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        acc = sum(l["acertou"] for l in linhas) / len(linhas)
        bts = sum(l["bytes"] for l in linhas) / len(linhas)
        rel.append(f"| {cond} | {acc:.1%} | {bts / 1e3:.1f} KB |")
        print(f"[orq2] {cond}: acurácia {acc:.1%}")
    (DIR_RESULTADOS / "relatorio_exp2.md").write_text("\n".join(rel) + "\n",
                                                     encoding="utf-8")
    print("\n".join(rel))


if __name__ == "__main__":
    main()
