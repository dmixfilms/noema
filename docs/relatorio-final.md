# Noema — Consolidated report, Experiments 0–4

*Qwen2.5-3B-Instruct (FP16, RTX 3090) | GSM8K test split, 50 problems | greedy
decoding, seed 42 | agents in separate processes, handoff via disk | Aug 2026*

## What was asked, and what was answered

**The thesis:** LLMs reason in a continuous vector space; text is a lossy
serialization of that state. Agents could communicate by transferring the raw
state (KV-cache) instead of text. Five experiments tested where that holds.

| # | Question | Result | Verdict |
|---|---|---|---|
| 0 | Does the channel exist? | L 56% = T 54%, Z 0%, KL = 0 | ✅ exists, perfect fidelity |
| 1 | How much of the cache is essential? | int4+last24: 56% at 17% of the bytes | ✅ 6× compression for free |
| 0.5 | Can one vector carry a thought? | 0%, total degeneration | ❌ not without training the receiver |
| 2 | Can different models understand each other? | ceiling 6%, bridge 2%, control 0% | ❌ distillation is the bottleneck |
| 4 | Does a real pipeline benefit? | L 46% = T 46% with 0 tokens between agents | ✅ parity, zero retransmission |

## Experiment 0 — Latent Handoff

Agent B, receiving only A's KV-cache + the suffix "Final answer:", solved 56%
of the problems — matching the text channel (54%) — **without receiving a single
word about the problem**. Negative control at 0% (no leakage). Fidelity:
KL = 0.000 between the next-token distribution of A-had-it-continued and of
B-with-the-cache-from-disk — B literally thinks from where A stopped. Cost:
9.4 MB of cache; handoff 0.022 s versus 0.087 s of text re-prefill.

## Experiment 1 — Wire Format

Bytes × accuracy curve along three compression axes, over the same 50 caches:

- **Quantization — the (almost) free axis:** int8 = 56% at 50% of the bytes
  (KL 0.002: the same thought); int4 = 54% at 25% (KL 2.1: a *different* thought
  that reaches the same result — reasoning is robust to state perturbations).
- **Temporal window — collapse:** keeping only the last m tokens discards the
  problem statement (which lives at the beginning) → 2–12% across all windows.
  In this short-context regime, the essential information is not at the end.
- **Layers — an asymmetric ramp:** the 12 shallow layers are expendable
  (last 24: 64%); the middle ones are critical (18: 44%; 12: 10%).
- **Best point: int4 + last 24 layers = 56% (same as baseline) at 1.59 MB —
  17% of the bytes.** 6× compression with no measurable loss.

## Experiment 0.5 — Continuous thought (homemade Coconut)

Transferring only the final hidden states (~4–18 KB) and injecting them as
`inputs_embeds`: 0% and degenerate text. Hidden states live outside the input
embedding distribution — which is why Coconut (Meta) *trains* the model to
consume its own states. An expected, documented null.

## Experiment 2 — Interlingua (3B → 1.5B)

A ridge adapter decoding A's states into B's "soft words".
v0.1/v0.2 (trained on plain text): nulls — domain shift.
v0.3 (trained on real reasoning states, K=16): **ceiling 6% | bridge 2% |
control 0%** — off the floor (coherent, on-domain answers) but weak.
The decisive number is the ceiling: even *within the same model*, distilling
thought into 16 vectors loses almost everything. The bottleneck is not the
cross-model crossing — it is **dimensional distillation**. Crossing that
frontier requires training the receiving model (order of effort: GPU-days),
not just the translator.

## Experiment 4 — The pipeline (extractor → calculator → verifier)

A realistic 3-agent pipeline (same checkpoint, different roles), two channels:

- **v1 — finding:** injecting the raw role instruction into the inherited
  stream degrades it (L 26% × T 42%). Continuation ≠ redirection: redirecting
  an inherited state requires the model's turn grammar.
- **v2 — structured turn:** the instruction enters as a complete template turn
  (fixed protocol tokens). Result: **L 46.0% = T 46.0%**, with **0 content
  tokens retransmitted** versus 407 on the text channel.
- Latency at short context: a technical tie (227 ms × 197 ms per pipeline) —
  disk handoff eats the prefill gain. Text prefill grows with context
  (80→94 ms along the pipeline); latent stays constant. The time advantage
  materializes at long contexts and/or with the Exp 1 wire format (which would
  shrink the pipeline's 19.4 MB to ~3.3 MB).

## Conclusions

1. **The noetic channel exists and is robust** when the state travels whole
   (or compressed up to ~6×): perfect fidelity, zero text cost.
2. **Thought does not survive dimensional distillation** with shallow
   adapters: from 9.4 MB to 50 KB, accuracy falls from 56% to 6%. The research
   frontier runs exactly there.
3. **Redirecting an inherited state** (giving a new role to whoever inherited
   the thought) works, but only within the model's conversation grammar.
4. **Security:** the latent channel is NOT encryption — the decoder (the
   model) is public. Protection comes from classical cryptography, perimeter
   zoning and a symbolic audit layer (see docs/ideia-roteador-cascata.md).
5. **Immediate application:** local multi-agent pipelines with the same
   checkpoint (the common case), exchanging compressed cache — zero
   retransmission, quality parity, constant per-hop cost. Candidate product:
   the cascade router (docs/ideia-roteador-cascata.md).

## Reproduction

Each experiment has its own orchestrator and an offline mechanical test:

```
noema_exp0/run_experiment.py [--smoke] [--kl]     # the channel
noema_exp1/run_exp1.py + fidelidade_exp1.py       # the compression
noema_exp05/run_exp05.py                          # continuous thought
noema_exp2/run_exp2.py                            # the interlingua
noema_exp4/run_exp4.py [--smoke]                  # the pipeline
```

Everything greedy with seed 42; raw results (JSONL per problem/condition) in
each experiment's resultados/ folder.
