"""Configuração central do Experimento 0 — Handoff Latente."""
import os
from pathlib import Path

import torch

# Modelo alvo do experimento (RTX 3090): Qwen2.5-3B-Instruct em FP16.
# Sobrescrevível via env p/ smoke test em máquinas sem GPU
# (ex.: NOEMA_MODEL=Qwen/Qwen2.5-0.5B-Instruct).
MODEL = os.environ.get("NOEMA_MODEL", "Qwen/Qwen2.5-3B-Instruct")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
N_PROBLEMAS = int(os.environ.get("NOEMA_N_PROBLEMAS", "50"))
CUT_RATIO = 0.6          # corte do raciocínio: 60% do comprimento médio de um CoT completo

N_PILOTO = 5             # problemas usados na calibração de N
MAX_COT_PILOTO = 512     # teto de tokens p/ um CoT completo no piloto
N_MIN, N_MAX = 32, 256   # clamp do N de corte
MAX_NEW_B = 48           # tokens que B pode gerar

SUFIXO = "\nResposta final:"

RAIZ = Path(__file__).resolve().parent
DIR_RESULTADOS = RAIZ / "resultados"
DIR_CACHES = RAIZ / "caches"

# id, pergunta, gold — lido só pelo orquestrador, pelo agente A e pela correção.
ARQ_PROBLEMAS = DIR_RESULTADOS / "problemas.jsonl"
# Canal textual A→B (condição T): id, pergunta, texto_parcial. Sem gold.
ARQ_HANDOFF_TEXTO = DIR_RESULTADOS / "handoff_texto.jsonl"
# Canal latente A→B (condição L): metadados dos caches. B nunca abre ARQ_PROBLEMAS.
ARQ_MANIFEST = DIR_CACHES / "manifest.json"
