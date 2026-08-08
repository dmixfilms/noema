# Experimento 2 v0.3 — Interlíngua: relatório

A (pensa): `Qwen/Qwen2.5-3B-Instruct` → B (conclui): `Qwen/Qwen2.5-1.5B-Instruct` | K=16 palavras-suaves | adaptador ridge treinado em estados de raciocínio reais (150 problemas do GSM8K train)

| Condição | Acurácia | Bytes/handoff |
|---|---|---|
| ponte | 2.0% | 65.5 KB |
| controle | 0.0% | 65.5 KB |
| teto | 6.0% | 49.2 KB |
