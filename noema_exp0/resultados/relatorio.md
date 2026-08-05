# Experimento 0 — Handoff Latente: relatório

Modelo: `Qwen/Qwen2.5-3B-Instruct` | device: `cuda` | seed: 42 | geração greedy | corte: 60% do CoT médio

| Condição | Acurácia | Acurácia (sem `concluiu_antes_do_corte`) | Tokens texto A→B (média) | Bytes cache (média) | Latência handoff (s) |
|---|---|---|---|---|---|
| T (textual) | 54.0% (50) | 54.0% (50) | 217 | 0.00e+00 | 0.087 |
| L (latente) | 56.0% (50) | 56.0% (50) | 0 | 9.37e+06 | 0.022 |
| Z (controle) | 0.0% (50) | 0.0% (50) | 0 | 0.00e+00 | — |

![gráfico](grafico.png)

## Leitura dos números

**H1 (confirmada).** A condição latente atingiu 56.0% de acurácia contra 54.0% da via textual. O agente B, sem receber uma única palavra sobre o problema, concluiu o raciocínio a partir do estado interno herdado de A — o desempenho da via latente iguala ou supera a via textual clássica.

**H2 (confirmada por construção, custo medido).** Na via latente trafegaram 0 tokens de texto entre A e B; toda a informação viajou nos 9.4 MB médios de KV-cache serializado, com handoff (desserialização + carga) de 0.022 s em média. Na via textual, A precisou transmitir 217 tokens em média e B pagou 0.087 s de re-prefill para reconstruir o contexto do zero. O cache bruto é o teto de custo em bytes: comprimi-lo é o objeto do Experimento 1.

**H3 (confirmada).** O controle negativo marcou 0.0%: o sufixo sozinho não resolve nada, provando que a informação está no estado latente transferido, não no prompt.

**Fidelidade da transferência.** KL médio entre a distribuição do próximo token de A (se continuasse) e de B (com o cache reidratado do disco): 0.000e+00. Valor ≈ 0 confirma que B literalmente pensa de onde A parou.
