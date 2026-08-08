"""Experimento 2 v0.3 — Interlíngua: handoff entre modelos DIFERENTES.

O agente A (Qwen2.5-3B) pensa; o agente B (Qwen2.5-1.5B — outro modelo, outras
dimensões) conclui, recebendo K "palavras-suaves": os últimos K hidden states
de A decodificados para o espaço de EMBEDDINGS de B por um adaptador ridge.

Lições das v0.1/v0.2 (nulos documentados em resultados/): hidden states crus
degeneram; adaptador treinado em texto corrido sofre desvio de domínio (estados
de raciocínio sob chat template vivem noutra região). A v0.3 corrige:
  - treino NO DOMÍNIO: os modelos raciocinam (chat template, greedy) sobre
    problemas do GSM8K train, e os pares são (estado no passo t → embedding do
    token emitido em t), colhidos ao longo da geração inteira;
  - canal mais largo: K=16 estados (era 4).

Condições na avaliação (50 problemas do test):
  ponte     — h_A × W_ponte → B          (a tese)
  controle  — h_A × W_aleatória → B      (piso de ruído)
  teto      — h_B próprios × W_self → B  (limite superior deste canal)

Fases em subprocessos (1 modelo por vez na VRAM):
  python run_exp2.py            # tudo: coleta A → coleta B → treina → avalia
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
N_PROBLEMAS_TREINO = int(os.environ.get("NOEMA_N_TREINO", "150"))
K_ESTADOS = int(os.environ.get("NOEMA_K", "16"))
AQUECIMENTO = int(os.environ.get("NOEMA_AQUECIMENTO", "8"))   # pula o início do CoT
STRIDE_GEN = int(os.environ.get("NOEMA_STRIDE", "4"))         # 1 par a cada N passos
RIDGE_LAMBDA = 10.0
SEED = 42


def _problemas_treino():
    """Perguntas de treino: GSM8K train (ou jsonl local via NOEMA_TREINO_LOCAL)."""
    local = os.environ.get("NOEMA_TREINO_LOCAL")
    if local:
        import metrics
        return [p["pergunta"] for p in metrics.ler_jsonl(local)][:N_PROBLEMAS_TREINO]
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    return [ds[i]["question"] for i in range(N_PROBLEMAS_TREINO)]


def _n_corte():
    import config
    manifest = json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))
    return manifest[0]["n_corte"]


@torch.no_grad()
def _pensar_colhendo(tok, model, pergunta: str, n_corte: int):
    """Gera o CoT greedy cortado em n_corte, colhendo pares
    (hidden state no passo t, id do token emitido em t). Retorna também os
    últimos K_ESTADOS hidden states (p/ avaliação)."""
    import config
    from transformers import DynamicCache
    msgs = [{"role": "user", "content": pergunta + "\nPense passo a passo."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt").to(config.DEVICE)
    cache = DynamicCache()

    def passo(x):
        out = model(x, past_key_values=cache, use_cache=True,
                    output_hidden_states=True,
                    attention_mask=torch.ones(1, cache.get_seq_length() + x.shape[1],
                                              dtype=torch.long, device=x.device))
        return (out.hidden_states[-1][:, -1].to(torch.float32).cpu(),
                out.logits[:, -1].argmax(-1, keepdim=True))

    pares_h, pares_id, ultimos = [], [], []
    h, nid = passo(ids)
    for t in range(n_corte):
        if nid.item() == tok.eos_token_id:
            break
        if t >= AQUECIMENTO and t % STRIDE_GEN == 0:
            pares_h.append(h)
            pares_id.append(nid.item())
        ultimos.append(h)
        del ultimos[:-K_ESTADOS]
        h, nid = passo(nid)
    return pares_h, pares_id, ultimos


@torch.no_grad()
def coletar(lado: str):
    """O modelo raciocina sobre os problemas de treino, no MESMO regime da
    avaliação. Lado B também computa os alvos de embedding (os ids de A foram
    salvos na fase A; tokenizer compartilhado)."""
    import nucleo
    tok, model = nucleo.carregar_modelo()
    n_corte = _n_corte()

    todos_h, todos_id = [], []
    for i, pergunta in enumerate(_problemas_treino()):
        pares_h, pares_id, _ = _pensar_colhendo(tok, model, pergunta, n_corte)
        todos_h += pares_h
        todos_id += pares_id
        if i % 25 == 0:
            print(f"[coleta/{lado}] {i}/{N_PROBLEMAS_TREINO} "
                  f"({len(todos_h)} pares)", flush=True)

    DIR_DADOS.mkdir(exist_ok=True)
    dados = {"estados": torch.cat(todos_h), "ids": todos_id}
    if lado == "B":
        emb = model.get_input_embeddings()
        ids_b = torch.tensor(todos_id, device=model.device)
        dados["alvos_self"] = emb(ids_b).to(torch.float32).cpu()
        ids_a = torch.tensor(
            torch.load(DIR_DADOS / "coleta_A.pt", weights_only=True)["ids"],
            device=model.device)
        dados["alvos_ponte"] = emb(ids_a).to(torch.float32).cpu()
    torch.save(dados, DIR_DADOS / f"coleta_{lado}.pt")
    print(f"[coleta/{lado}] {len(todos_h)} pares salvos", flush=True)


def _ridge(X, Y):
    Xa = torch.cat([X, torch.ones(len(X), 1)], dim=1)
    XtX = Xa.T @ Xa + RIDGE_LAMBDA * torch.eye(Xa.shape[1])
    W = torch.linalg.solve(XtX, Xa.T @ Y)
    residuo = ((Xa @ W - Y).pow(2).mean().sqrt() / Y.pow(2).mean().sqrt()).item()
    return W, residuo


def treinar():
    """W_ponte: h_A(t) → emb_B(token emitido por A em t) — decodifica o
    pensamento de A em palavras-suaves de B. W_self: idem dentro de B."""
    torch.manual_seed(SEED)
    a = torch.load(DIR_DADOS / "coleta_A.pt", weights_only=True)
    b = torch.load(DIR_DADOS / "coleta_B.pt", weights_only=True)
    W_ponte, r1 = _ridge(a["estados"], b["alvos_ponte"])
    W_self, r2 = _ridge(b["estados"], b["alvos_self"])
    torch.save({"W_ponte": W_ponte, "W_self": W_self}, DIR_DADOS / "adaptador.pt")
    print(f"[treino] {len(a['estados'])} pares | W_ponte {tuple(W_ponte.shape)} "
          f"(erro rel. {r1:.3f}) | W_self {tuple(W_self.shape)} (erro rel. {r2:.3f})")


@torch.no_grad()
def avaliar_a():
    """A pensa nos 50 problemas do test e exporta os últimos K hidden states."""
    import config
    import metrics
    import nucleo
    tok, model = nucleo.carregar_modelo()
    n_corte = _n_corte()
    saida = {}
    for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS):
        _, _, ultimos = _pensar_colhendo(tok, model, p["pergunta"], n_corte)
        saida[p["id"]] = torch.cat(ultimos)  # [K, dim_a]
        print(f"[aval/A] problema {p['id']} pensado", flush=True)
    DIR_DADOS.mkdir(exist_ok=True)
    torch.save(saida, DIR_DADOS / "eval_estados_A.pt")


@torch.no_grad()
def avaliar_b(cond: str):
    """B (o receptor) conclui a partir das palavras-suaves + sufixo."""
    import config
    import metrics
    import nucleo
    from transformers import DynamicCache
    tok, model = nucleo.carregar_modelo()
    n_corte = _n_corte()
    emb = model.get_input_embeddings()
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

    linhas = []
    for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS):
        if cond == "teto":
            _, _, ultimos = _pensar_colhendo(tok, model, p["pergunta"], n_corte)
            h_orig = torch.cat(ultimos)          # [K, dim_b]
            Wx = W_self
        else:
            h_orig = estados_a[p["id"]]          # [K, dim_a]
            Wx = W
        X = torch.cat([h_orig, torch.ones(h_orig.shape[0], 1)], dim=1)
        h = (X @ Wx).unsqueeze(0).to(config.DEVICE, config.DTYPE)

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
        linhas.append({"id": p["id"], "condicao": cond, "resposta_gerada": texto,
                       "bytes": h_orig.numel() * 2,
                       "latencia_total_s": time.time() - t0})
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

    print(f"[orq2] v0.3 | A = {MODEL_A}\n[orq2] B = {MODEL_B} | K = {K_ESTADOS}",
          flush=True)
    _sub("--coletar", MODEL_A, ("A",))
    _sub("--coletar", MODEL_B, ("B",))
    _sub("--treinar", MODEL_B)
    _sub("--avaliar-a", MODEL_A)
    for cond in ("ponte", "controle", "teto"):
        _sub("--avaliar-b", MODEL_B, (cond,))

    gold = {p["id"]: p["gold"] for p in metrics.ler_jsonl(config.ARQ_PROBLEMAS)}
    rel = ["# Experimento 2 v0.3 — Interlíngua: relatório", "",
           f"A (pensa): `{MODEL_A}` → B (conclui): `{MODEL_B}` | K={K_ESTADOS} "
           f"palavras-suaves | adaptador ridge treinado em estados de raciocínio "
           f"reais ({N_PROBLEMAS_TREINO} problemas do GSM8K train)", "",
           "| Condição | Acurácia | Bytes/handoff |", "|---|---|---|"]
    for cond in ("ponte", "controle", "teto"):
        linhas = metrics.ler_jsonl(DIR_RESULTADOS / f"brutos_{cond}.jsonl")
        with open(DIR_RESULTADOS / f"resultados_{cond}.jsonl", "w",
                  encoding="utf-8") as f:
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
