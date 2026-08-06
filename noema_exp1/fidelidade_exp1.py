"""Fidelidade do wire format: KL(cache cheio || cache comprimido) por configuração.

Para 10 problemas: a distribuição do próximo token de B com o cache cheio é a
referência; cada configuração de compressão gera a sua, e o KL mede quanto o
pensamento "desvia" sob compressão — acurácia diz se acerta, KL diz o quão
diferente pensa.

Uso: python fidelidade_exp1.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noema_exp0"))
import config      # noqa: E402
import nucleo      # noqa: E402
import wire        # noqa: E402
from run_exp1 import CONFIGS  # noqa: E402

DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"
N_PROBLEMAS_KL = 10


@torch.no_grad()
def main():
    torch.manual_seed(config.SEED)
    tok, model = nucleo.carregar_modelo()
    manifest = json.load(open(config.ARQ_MANIFEST, encoding="utf-8"))[:N_PROBLEMAS_KL]
    ids_sufixo = tok(config.SUFIXO, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(config.DEVICE)

    soma = {c["tag"]: 0.0 for c in CONFIGS}
    for m in manifest:
        payload = torch.load(config.DIR_CACHES / m["cache_file"],
                             map_location="cpu", weights_only=True)

        # referência: cache cheio (reconstruído a cada uso — _passo muta o cache)
        cheio = wire.comprimir(payload["k"], payload["v"], quant="fp16")
        cache_ref, pos_ref = wire.reidratar(cheio, config.DEVICE, config.DTYPE)
        logits_ref, _ = nucleo._passo(model, ids_sufixo, cache_ref, pos=pos_ref)
        log_p = F.log_softmax(logits_ref.float(), dim=-1)

        for cfg in CONFIGS:
            pacote = wire.comprimir(payload["k"], payload["v"],
                                    quant=cfg.get("quant", "fp16"),
                                    janela=cfg.get("janela"),
                                    camadas=cfg.get("camadas"))
            cache_c, pos_c = wire.reidratar(pacote, config.DEVICE, config.DTYPE)
            logits_c, _ = nucleo._passo(model, ids_sufixo, cache_c, pos=pos_c)
            log_q = F.log_softmax(logits_c.float(), dim=-1)
            soma[cfg["tag"]] += F.kl_div(log_q, log_p, log_target=True,
                                         reduction="sum").item()
        print(f"[KL1] problema {m['id']} processado", flush=True)

    medias = {tag: s / len(manifest) for tag, s in soma.items()}
    DIR_RESULTADOS.mkdir(exist_ok=True)
    with open(DIR_RESULTADOS / "fidelidade_exp1.json", "w", encoding="utf-8") as f:
        json.dump(medias, f, indent=2)
    for tag, kl in sorted(medias.items(), key=lambda x: x[1]):
        print(f"[KL1] {tag}: KL médio = {kl:.4f}")


if __name__ == "__main__":
    main()
