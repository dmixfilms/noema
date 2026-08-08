# Idea: Cascade router with thought-based escalation

*Recorded 2026-08-08, during Experiments 0–2. Origin: Anderson.*

## The product thesis

"This is what people need today": every serious chatbot faces the same
dilemma — a light model answers fast and cheap but fails on hard cases; a
heavy model gets it right but is too expensive and slow to be the front door.

The architecture:

```
[user] → LIGHT model (front door: fast, precise, runs on any machine)
           │ self-detection: measures its own uncertainty on every answer
           │ (entropy/logprobs from token-by-token generation — mechanism already built)
           ├─ confident → answers immediately (most cases)
           └─ uncertain / needs deep database work / images →
                ESCALATES to the specialist model (bigger, multimodal,
                database access, smarter)
```

**The Noema differentiator:** escalation doesn't travel as tokens — it travels
as **thought** (latent state). The big model doesn't re-read the conversation
from scratch: it inherits the understanding the light model already formed and
continues from there.

## Why thought-based escalation, if local tokens are "free"

A local token costs no money per call, but it costs three real things:

1. **Latency** — with text escalation the big model pays the re-prefill of the
   entire context. Measured in Exp 0: latent handoff 0.022 s vs 0.087 s of
   re-prefill at ~260 tokens — and re-prefill grows linearly with context (at
   20k tokens, seconds versus disk I/O).
2. **GPU capacity (= money in a company)** — every re-prefill burns FLOPs that
   could serve another user. Less reprocessing = more users per card.
3. **Fidelity** — text is lossy serialization: what the light model already
   understood (resolved ambiguities, user context, entities matched against the
   database) gets flattened and rebuilt differently by the big model. Latent
   escalation transfers the understanding, not the transcript (Exp 0: KL = 0).

## Security note (important — corrects a common intuition)

The latent channel is NOT encryption. The KV-cache contains the data in full
(Exp 0: KL = 0, nothing is lost) and the decoder is public: anyone with the
file + the (open) checkpoint extracts the content — Exp 0's own agente_b.py is
the extraction tool. Leakage level ≈ plaintext against a competent attacker,
with the aggravating factors of a false sense of security and of blinding audit
tooling (DLP/logs don't inspect tensors). Policies trained into the model don't
protect the cache: it can be probed without generating text (linear probing).

What actually protects: classical encryption on the channel and at rest, trust
zones (sensitive cache never crosses the perimeter) and the Exp 3 contract
layer — mandatory precisely because the channel is unauditable. The correct
pitch is "faster, more faithful, cheaper, with classical security and
auditability through the symbolic layer" — never "secure because unreadable".

## State of the art (internal)

- Escalation between copies of the SAME model: works today (Exp 0/1; wire
  format int4+last24 = 17% of the bytes with no loss).
- Escalation between DIFFERENT models (light→heavy, text→vision): requires the
  Exp 2 adapter — v0.3 inconclusive (ceiling 6% / bridge 2%); the bottleneck is
  dimensional distillation; next step would be training the receiving model
  (Coconut-style), an order-of-magnitude bigger effort.
- Hybrid version buildable NOW: router with entropy-based self-detection +
  text escalation, with a ready "slot" to swap the boundary to latent when
  Exp 2 matures. The hybrid router also produces the baseline that measures
  how much the text boundary loses.
