# Noema — Relatório consolidado dos Experimentos 0–4

*Qwen2.5-3B-Instruct (FP16, RTX 3090) | GSM8K test, 50 problemas | geração greedy,
seed 42 | agentes em processos separados, handoff via disco | ago/2026*

## O que foi perguntado e o que foi respondido

**A tese:** LLMs raciocinam em espaço vetorial contínuo; o texto é uma serialização
com perda desse estado. Agentes poderiam se comunicar transferindo o estado bruto
(KV-cache) em vez de texto. Cinco experimentos testaram onde isso funciona.

| # | Pergunta | Resultado | Veredito |
|---|---|---|---|
| 0 | O canal existe? | L 56% = T 54%, Z 0%, KL = 0 | ✅ existe, fidelidade perfeita |
| 1 | Quanto do cache é essencial? | int4+cam24: 56% com 17% dos bytes | ✅ compressão 6× de graça |
| 0.5 | Um vetor carrega pensamento? | 0%, degeneração total | ❌ não sem treinar o receptor |
| 2 | Modelos diferentes se entendem? | teto 6%, ponte 2%, controle 0% | ❌ destilação é o gargalo |
| 4 | Uma esteira real compensa? | L 46% = T 46% com 0 tokens entre agentes | ✅ paridade sem retransmissão |

## Experimento 0 — Handoff Latente

O agente B, recebendo apenas o KV-cache de A + o sufixo "Resposta final:",
resolveu 56% dos problemas — empatado com a via textual (54%) — **sem receber uma
palavra sobre o problema**. Controle negativo em 0% (sem vazamento). Fidelidade:
KL = 0,000 entre a distribuição de próximo token de A-se-continuasse e de
B-com-cache-do-disco — B literalmente pensa de onde A parou. Custo: 9,4 MB de
cache; handoff 0,022 s contra 0,087 s de re-prefill textual.

## Experimento 1 — Wire Format

Curva bytes × acurácia em três eixos de compressão, sobre os mesmos 50 caches:

- **Quantização — o eixo quase grátis:** int8 = 56% com 50% dos bytes (KL 0,002:
  o mesmo pensamento); int4 = 54% com 25% (KL 2,1: pensamento *diferente* que
  chega ao mesmo resultado — o raciocínio é robusto a perturbações no estado).
- **Janela temporal — colapso:** manter só os últimos m tokens descarta o
  enunciado (que mora no início) → 2–12% em todas as janelas. Neste regime de
  contexto curto, a informação essencial não está no fim.
- **Camadas — rampa assimétrica:** as 12 camadas rasas são dispensáveis
  (24 finais: 64%); as médias são críticas (18: 44%; 12: 10%).
- **Melhor ponto: int4 + últimas 24 camadas = 56% (igual ao baseline) com 1,59 MB
  — 17% dos bytes.** Compressão 6× sem perda mensurável.

## Experimento 0.5 — Pensamento contínuo (Coconut caseiro)

Transferir só os hidden states finais (~4–18 KB) e injetá-los como
`inputs_embeds`: 0% e texto degenerado. Hidden states vivem fora da distribuição
dos embeddings de entrada — é por isso que o Coconut (Meta) *treina* o modelo
para consumir os próprios estados. Nulo esperado e documentado.

## Experimento 2 — Interlíngua (3B → 1.5B)

Adaptador ridge decodificando estados de A em "palavras-suaves" de B.
v0.1/v0.2 (treino em texto corrido): nulos — desvio de domínio.
v0.3 (treino em estados de raciocínio reais, K=16): **teto 6% | ponte 2% |
controle 0%** — saiu do zero (respostas coerentes, no domínio), mas fraco.
O dado decisivo é o teto: mesmo *dentro do mesmo modelo*, destilar o pensamento
em 16 vetores perde quase tudo. O gargalo não é a travessia entre modelos — é a
**destilação dimensional**. Atravessar essa fronteira exige treinar o modelo
receptor (ordem de esforço: dias de GPU), não apenas o tradutor.

## Experimento 4 — A esteira (extrator → calculador → verificador)

Pipeline realista de 3 agentes (mesmo checkpoint, papéis diferentes), duas vias:

- **v1 — achado:** injetar a instrução de papel crua no fluxo herdado degrada
  (L 26% × T 42%). Continuação ≠ redirecionamento: redirecionar um estado
  herdado exige a gramática de turnos do modelo.
- **v2 — turno estruturado:** a instrução entra como turno completo do template
  (tokens fixos de protocolo). Resultado: **L 46,0% = T 46,0%**, com **0 tokens
  de conteúdo retransmitidos** contra 407 da via textual.
- Latência em contexto curto: empate técnico (227 ms × 197 ms por esteira) — o
  handoff de disco come o ganho de prefill. O prefill textual cresce com o
  contexto (80→94 ms ao longo da esteira); o latente é constante. A vantagem de
  tempo se materializa em contextos longos e/ou com o wire format do Exp 1
  (reduziria os 19,4 MB da esteira a ~3,3 MB).

## Conclusões

1. **O canal noético existe e é robusto** quando o estado viaja inteiro (ou
   comprimido até ~6×): fidelidade perfeita, custo textual zero.
2. **O pensamento não sobrevive à destilação dimensional** com adaptadores
   rasos: de 9,4 MB para 50 KB, a acurácia cai de 56% para 6%. A fronteira da
   pesquisa passa exatamente aí.
3. **Redirecionar um estado herdado** (dar novo papel a quem herdou o
   pensamento) funciona, mas só dentro da gramática de conversa do modelo.
4. **Segurança:** o canal latente NÃO é criptografia — o decodificador (o
   modelo) é público. Proteção vem de criptografia clássica, perímetro e camada
   de auditoria simbólica (ver docs/ideia-roteador-cascata.md).
5. **Aplicação imediata:** esteiras multi-agente locais com o mesmo checkpoint
   (o caso comum), trocando cache comprimido — zero retransmissão, paridade de
   qualidade, custo de salto constante. Produto candidato: o roteador em
   cascata (docs/ideia-roteador-cascata.md).

## Reprodução

Cada experimento tem orquestrador próprio e teste mecânico offline:

```
noema_exp0/run_experiment.py [--smoke] [--kl]     # o canal
noema_exp1/run_exp1.py + fidelidade_exp1.py       # a compressão
noema_exp05/run_exp05.py                          # o pensamento contínuo
noema_exp2/run_exp2.py                            # a interlíngua
noema_exp4/run_exp4.py [--smoke]                  # a esteira
```

Tudo greedy com seed 42; resultados brutos (JSONL por problema/condição) nas
pastas resultados/ de cada experimento.
