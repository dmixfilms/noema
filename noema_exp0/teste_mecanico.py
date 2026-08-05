"""Smoke test MECÂNICO do protocolo, sem rede.

Em ambientes sem acesso ao HuggingFace Hub, valida toda a mecânica do
Experimento 0 com um modelo Qwen2 minúsculo de pesos aleatórios e um tokenizer
BPE treinado localmente: geração token a token, serialização/reidratação do
KV-cache entre processos, as três condições, correção, fidelidade (KL) e
relatório. NÃO valida acurácia (o modelo é aleatório) — valida o instrumento.

Uso: python teste_mecanico.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_MODELO = RAIZ / "modelo_teste"


def construir_modelo_teste():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)

    corpus = [
        "Quanto é 2 + 2? Pense passo a passo. Resposta final: 4",
        "Maria tem 3 maçãs e ganha 5. Total de 8 maçãs.",
        "Um trem anda 60 km em 1 hora. Em 2 horas anda 120 km.",
        "0 1 2 3 4 5 6 7 8 9 10 100 1000",
    ] * 50
    bpe = Tokenizer(models.BPE(unk_token="<|unk|>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=["<|unk|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|end|>"],
    )
    bpe.train_from_iterator(corpus, trainer)

    tok = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|eos|>",
    )
    tok.chat_template = (
        "{% for message in messages %}<|user|>{{ message['content'] }}<|end|>"
        "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )

    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    modelo = Qwen2ForCausalLM(cfg)

    DIR_MODELO.mkdir(exist_ok=True)
    tok.save_pretrained(DIR_MODELO)
    modelo.save_pretrained(DIR_MODELO)
    print(f"[teste] modelo aleatório ({sum(p.numel() for p in modelo.parameters()):,} "
          f"parâmetros) salvo em {DIR_MODELO}")


def preparar_problemas_fake():
    resultados = RAIZ / "resultados"
    resultados.mkdir(exist_ok=True)
    problemas = [
        {"id": 0, "pergunta": "Quanto é 2 + 2?", "gold": 4},
        {"id": 1, "pergunta": "Maria tem 3 maçãs e ganha 5. Quantas tem?", "gold": 8},
        {"id": 2, "pergunta": "Um trem anda 60 km/h. Quanto anda em 2 horas?", "gold": 120},
    ]
    with open(resultados / "problemas.jsonl", "w", encoding="utf-8") as f:
        for p in problemas:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    construir_modelo_teste()
    preparar_problemas_fake()

    env = dict(os.environ, NOEMA_MODEL=str(DIR_MODELO))
    subprocess.run(
        [sys.executable, str(RAIZ / "run_experiment.py"),
         "--problemas-prontos", "--n-corte", "16", "--kl"],
        check=True, cwd=RAIZ, env=env,
    )

    # Verificações mecânicas
    import metrics
    falhas = []
    for cond in ["L", "T", "Z"]:
        linhas = metrics.ler_jsonl(RAIZ / "resultados" / f"resultados_{cond}.jsonl")
        if len(linhas) != 3:
            falhas.append(f"condição {cond}: esperava 3 linhas, veio {len(linhas)}")
    lat = metrics.ler_jsonl(RAIZ / "resultados" / "resultados_L.jsonl")
    if any(l["tokens_texto_A_para_B"] != 0 for l in lat):
        falhas.append("condição L trafegou tokens de texto (deveria ser 0)")
    if any(l["bytes_cache"] <= 0 for l in lat):
        falhas.append("condição L sem bytes de cache registrados")
    txt = metrics.ler_jsonl(RAIZ / "resultados" / "resultados_T.jsonl")
    if any(l["tokens_texto_A_para_B"] <= 0 for l in txt):
        falhas.append("condição T sem tokens de texto registrados")
    kl = json.load(open(RAIZ / "resultados" / "fidelidade_kl.json", encoding="utf-8"))
    if abs(kl["kl_medio"]) > 1e-3:
        falhas.append(f"KL médio {kl['kl_medio']:.3e} > 1e-3: handoff não é fiel")

    if falhas:
        print("\n[teste] FALHAS:")
        for f in falhas:
            print("  -", f)
        raise SystemExit(1)
    print(f"\n[teste] OK — protocolo mecânico validado "
          f"(KL médio do handoff: {kl['kl_medio']:.3e})")


if __name__ == "__main__":
    main()
