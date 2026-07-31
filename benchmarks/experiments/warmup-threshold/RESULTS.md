# Warm-up threshold experiment (issue #434)

**Question:** what is a good value for the TQ+ warm-up threshold `t`
(`TQPLUS_MIN_SAMPLES`, currently 1000), and for the cap of the proposed
capped exponential-backoff recalibration schedule (refit at `t, 2t, 4t, …`
up to a cap)?

**Method:** the real turbovec encode/search path (release wheel), with an
experiment-only env override (`TURBOVEC_TQPLUS_MIN_SAMPLES`) that lets the
first `add()` pin the calibration fit to an exact sample size. All later
rows are added under the frozen calibration — identical to the state of a
saved-and-reloaded index. `fit_n = 0` means identity calibration forever.
Quality metric: recall of the exact float32 top-10 in the quantized top-10
(`R@10`) and top-100 (`R10@100`), 1000 held-out queries against 100k
database vectors. Insertion orders: i.i.d. shuffles, sorted by first
principal component (`pc1`, worst-case ordered stream), sorted by norm
(`norm`, mild realistic bias).

Datasets: GloVe-200-angular (ann-benchmarks), OpenAI
text-embedding-3-large-1536 (Qdrant dbpedia 1M, first 101k rows),
synthetic 256-d cone (dominant mean + heavy-tailed low-rank factors).
Raw numbers in `results/`; reproduce with `run_experiment.py`.

## Findings

**1. For i.i.d. streams, fit size is irrelevant above ~100–200 samples.**
On every dataset and bit width, a calibration fitted from 100 rows is
statistically indistinguishable from one fitted from all 100k (R@10 within
±0.003, comparable to seed noise). Example, OpenAI-1536 2-bit shuffled:
0.8983 at fit 100 vs 0.8993 at fit 100k. The estimation-noise rationale for
`t = 1000` ("quantiles too noisy below it") is not visible in end recall at
all: quantile variance is a solved problem by n ≈ 100.

**2. Fit-sample *bias* is the actual risk, and it needs ~16k–32k samples to
heal.** With `pc1`-ordered insertion on OpenAI-1536 (2-bit), a fit from the
first 100 rows gives 0.8726 — *worse than identity's* 0.8946 — climbing
monotonically: 0.8821 at 1k, 0.8968 at 16k, 0.8991 at 32k. The same shape
appears at 3/4-bit and on the other datasets (GloVe norm-order 2-bit:
0.5537 R10@100 at fit 1k vs 0.5951 full-fit vs 0.6149 identity). Today's
behaviour — freeze forever at ~1000 rows — picks the worst point on this
curve for any non-shuffled stream.

**3. The net value of calibration is dataset-dependent and small.**
- OpenAI-1536 (best case, i.i.d. order): +0.005 R@10 at 2-bit
  (0.8946 → 0.8995), +0.002 at 3-bit, ≈ +0.001 at 4-bit.
- GloVe-200: calibration is a net *negative* at every fit size and bit
  width (2-bit R@10 0.2419 identity vs 0.2255 full-fit).
- Synthetic cone: strongly negative at 2-bit (R10@100 0.119 identity vs
  0.062 full-fit); a score-fidelity probe shows fitted calibration
  distorts returned scores badly there (corr with true inner product 0.58
  vs 0.96 for identity).
- Refitting with a wider quantile pair (1%/99% instead of 5%/95%) does not
  change this (GloVe 2-bit full-fit 0.2276 vs 0.2255), so it is not a
  tail-clipping tuning issue — the affine-to-canonical-Beta model itself
  simply does not beat identity + per-row scale on those distributions.
- With a rerank stage the entire effect vanishes on OpenAI-1536:
  R10@100 = 1.0000 in every configuration, fitted or identity.

## Recommendations

- **`t`: keep 1000** (anything in ~200–2000 is equivalent; 1000 changes no
  existing behaviour or encoded bytes for the first crossing). There is no
  recall payoff to tuning it, so it should not be a knob users must think
  about — expose it if desired, but the default needs no user input.
- **The knob that matters is the backoff cap: default `32t = 32 000`.**
  Refits at `t, 2t, …, 32t` are what heal biased streams (parity with the
  full fit by 16k–32k on every dataset tested); total re-encode work stays
  ≤ 2× by the geometric-series argument. Warm-up buffer cost until the cap:
  `32 000 × dim × 4 B` (≈ 26 MB at dim 200, ≈ 197 MB at dim 1536) —
  bounded and temporary. Serializing the buffer below the cap makes
  save/reload schedule-neutral at the price of that many extra bytes in
  the file.
- **On/off: make calibration an explicit constructor option.** Measured
  effect sizes are +0.5 pp R@10 at 2-bit on modern embeddings, ~0 at 4-bit
  or under rerank, and *negative* on GloVe-like data. Default-off is
  defensible on this evidence; default-on is only worth it once the
  calibration-loses-to-identity anomaly (finding 3) is understood — that
  deserves its own issue, since it questions what the warm-up machinery is
  protecting in the first place.
- **The warm-up warning and its latch (issue #434) become deletable** under
  backoff + buffer serialization: saving below the cap no longer forfeits
  anything, so there is nothing left to warn about.

## Reproduction

```
# build a wheel from this branch (contains the env override), then:
python run_experiment.py prep-glove     # or prep-openai / prep-synth
python run_experiment.py run ~/data/py-turboquant/warmup-threshold/glove_db.npy \
    ~/data/py-turboquant/warmup-threshold/glove_q.npy out.csv 2,3,4 \
    0,100,200,400,1000,2000,4000,8000,16000,32000,64000,100000
python run_experiment.py report out.csv
```
