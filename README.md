# Noema — Experimento 0: Handoff Latente

Primeiro experimento do projeto **Noema**: comunicação entre agentes de IA sem texto como
veículo do pensamento. O agente A raciocina sobre um problema do GSM8K, é interrompido no
meio, e transfere seu **estado interno bruto (KV-cache)** — via disco, entre processos
separados — para o agente B, que conclui o raciocínio **sem receber uma única palavra
sobre o problema**.

## Três condições

| Condição | O que B recebe | Custo medido |
|---|---|---|
| **T** (textual) | problema + raciocínio parcial de A, em texto | tokens de texto A→B |
| **L** (latente) | KV-cache de A serializado + sufixo `"\nResposta final:"` | bytes do cache; **0 tokens de texto** |
| **Z** (controle) | só o sufixo, sem cache e sem problema | — (acurácia esperada ≈ 0) |

Mesmo checkpoint, geração greedy, seed fixa — as condições são comparáveis por construção.

## Reprodução

Requisitos: Python 3.11, GPU CUDA com ≥8 GB VRAM para o modelo padrão
(`Qwen/Qwen2.5-3B-Instruct` em FP16; ~6.5 GB).

```bash
python -m venv noema_env
noema_env\Scripts\activate                 # Windows
# source noema_env/bin/activate            # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.46,<5" accelerate datasets matplotlib

cd noema_exp0
python run_experiment.py --smoke           # 1) smoke test: 3 problemas, N fixo=80
python run_experiment.py                   # 2) rodada completa: 50 problemas × 3 condições
python run_experiment.py --kl              #    (opcional) + métrica de fidelidade (KL)
```

Saídas em `noema_exp0/resultados/`: um `resultados_{T,L,Z}.jsonl` por condição,
`relatorio.md` com a tabela comparativa e `grafico.png`.

Sem GPU (só para validar o protocolo — lento e com modelo menor):

```bash
NOEMA_MODEL=Qwen/Qwen2.5-0.5B-Instruct python run_experiment.py --smoke
```

Modelos já baixados no cache local do HuggingFace (`~/.cache/huggingface`) são
reaproveitados automaticamente; `NOEMA_MODEL` também aceita um caminho local.

Sem rede nenhuma, há um smoke test mecânico que valida todo o protocolo (processos
separados, serialização do cache, 3 condições, KL) com um modelo minúsculo de pesos
aleatórios — não valida acurácia, valida o instrumento:

```bash
python teste_mecanico.py
```

## Estrutura

```
noema_exp0/
├── config.py          # modelo, n_problemas=50, cut_ratio=0.6, seed=42, paths
├── nucleo.py          # geração manual token a token + (de)serialização do DynamicCache
├── agente_a.py        # processo A: calibra N, pensa até o corte, serializa caches
├── agente_b.py        # processo B: conclui a partir de cache (L) / texto (T) / nada (Z)
├── run_experiment.py  # orquestra os subprocessos, corrige e agrega
├── metrics.py         # extração da resposta numérica GSM8K, acurácia, custos
├── report.py          # tabela final + gráfico (matplotlib)
└── resultados/        # JSONL por condição + relatorio.md
```

Detalhes de protocolo:

- **Separação real de processos**: `agente_a.py` e `agente_b.py` rodam como subprocessos
  independentes; o handoff latente passa por `caches/cache_problema_{i}.pt` no disco.
  Na condição L, o processo B **nunca abre o arquivo de problemas** — lê apenas o
  manifest de caches e o sufixo fixo.
- **Calibração do corte**: N = 60% do comprimento médio de um CoT completo, medido num
  piloto de 5 problemas (clamp em [32, 256]). Problemas em que A conclui antes do corte
  recebem a flag `concluiu_antes_do_corte` e são reportados separadamente.
- **Determinismo**: greedy em tudo, `torch.manual_seed(42)`.
- **Compatibilidade de API**: a (de)serialização do `DynamicCache` funciona tanto com a
  interface `key_cache/value_cache` quanto com a mais recente `cache.layers`, e a
  reconstrução usa `cache.update()` (estável entre versões de transformers).
