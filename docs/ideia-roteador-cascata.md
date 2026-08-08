# Ideia: Roteador em cascata com escalada por pensamento

*Registrada em 2026-08-08, durante os Experimentos 0–2. Origem: Anderson.*

## A tese de produto

"Essa ideia é a necessidade do povo hoje": todo chatbot sério tem o mesmo dilema —
modelo leve responde rápido e barato mas erra no difícil; modelo pesado acerta mas é
caro e lento demais pra ser a porta de entrada.

A arquitetura:

```
[usuário] → modelo LEVE (porta de entrada: rápido, preciso, roda em qualquer máquina)
              │ autodetecção: mede a própria incerteza a cada resposta
              │ (entropia/logprobs da geração token a token — já temos o mecanismo)
              ├─ confiante → responde na hora (maioria dos casos)
              └─ incerto / precisa de consulta profunda / imagem →
                   ESCALA para o modelo especialista (maior, multimodal,
                   com acesso a banco de dados, mais inteligente)
```

**O diferencial Noema:** a escalada não vai por tokens — vai por **pensamento**
(estado latente). O modelo grande não relê a conversa do zero: ele herda a
compreensão que o leve já formou e continua dali.

## Por que escalada por pensamento, se tokens locais são "grátis"

Token local não custa dinheiro por chamada, mas custa três coisas reais:

1. **Latência** — na escalada textual o modelo grande paga o re-prefill do contexto
   inteiro. Medido no Exp 0: handoff latente 0,022 s vs 0,087 s de re-prefill com
   ~260 tokens — e o re-prefill cresce linearmente com o contexto (em 20k tokens,
   segundos vs I/O de disco).
2. **Capacidade da GPU (= dinheiro em empresa)** — cada re-prefill queima FLOPs que
   poderiam atender outro usuário. Menos re-processamento = mais usuários por placa.
3. **Fidelidade** — o texto é serialização com perda: o que o leve já entendeu
   (ambiguidades resolvidas, contexto do usuário, entidades do banco) é achatado e
   reconstruído diferente pelo grande. A escalada latente transfere a compreensão,
   não a transcrição (Exp 0: KL = 0).

## Nota de segurança (importante — corrige uma intuição comum)

O canal latente NÃO é criptografia. O KV-cache contém os dados integralmente
(Exp 0: KL = 0, nada se perde) e o decodificador é público: qualquer pessoa com
o arquivo + o checkpoint (aberto) extrai o conteúdo — o próprio agente_b.py do
Exp 0 é a ferramenta de extração. Nível de vazamento ≈ texto puro para um
atacante competente, com o agravante de dar falsa sensação de segurança e de
cegar as ferramentas de auditoria (DLP/logs não inspecionam tensores). Políticas
treinadas no modelo não protegem o cache: ele pode ser sondado sem gerar texto
(linear probing).

O que protege de verdade: criptografia clássica no canal e em repouso, zonas de
confiança (cache sensível nunca cruza o perímetro) e a camada de contratos do
Exp 3 — obrigatória justamente porque o canal é inauditável. O pitch correto é
"mais rápido, mais fiel, mais barato, com segurança clássica e auditabilidade
pela camada simbólica" — nunca "seguro porque ilegível".

## Estado da técnica (interno)

- Escalada entre cópias do MESMO modelo: funciona hoje (Exp 0/1; wire format
  int4+cam24 = 17% dos bytes sem perda).
- Escalada entre modelos DIFERENTES (leve→pesado, texto→visão): exige o adaptador
  do Exp 2 — v0.1 inconclusiva (ponte 4% vs teto quebrado por desvio de domínio);
  próximos passos: treinar o adaptador em estados de raciocínio sob chat template
  (mesmo domínio da inferência), k maior, MLP se o linear não bastar.
- Versão híbrida construível JÁ: roteador com autodetecção por entropia + escalada
  textual, com "slot" pronto para trocar a fronteira por latente quando o Exp 2
  amadurecer. O roteador híbrido também gera o baseline que mede quanto a fronteira
  textual perde.
