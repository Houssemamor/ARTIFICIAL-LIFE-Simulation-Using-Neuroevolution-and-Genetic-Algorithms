# Phase 1.5 Benchmark Report -- Batched Population Inference

**Date:** 2026-09-20
**Environment:** Windows, Python 3.14.5, torch 2.14.0+cpu, numpy 2.5.3 (CPU only)
**Command:** `python -m pytest tests/performance/test_batched_inference_benchmark.py -s`
**Architecture:** 12-32-16-3 fixed controller (configs/baseline.json)

## Measured Steps/Second

| Population | Batched (steps/s) | Naive per-agent (steps/s) | Speedup |
|---|---|---|---|
| 50  | 12,082.0 | 95.02  | 127.2x |
| 100 | 14,159.1 | 48.32  | 293.1x |
| 250 | 10,762.2 | 19.33  | 556.7x |
| 500 |  7,830.5 |  9.59  | 816.3x |

## Exit-Criterion Check

- Real-time target: 30 FPS (the app's declared render rate, app/run.py:64).
- Batched at population 250: **10,762 steps/s** -- 358x the target, ~45,000x
  budget per 250-agent physics step. Real-time NFR confirmed achievable.
- The naive path at 250 agents (19 steps/s) falls below real time, which is
  the exact reason the batched path exists and must ship. Quarantined in
  `neural/_naive_inference.py`; never imported from production modules.

## Correctness Gate

- `test_batched_matches_naive`: batched output equals the per-agent reference
  within 1e-5 float tolerance at populations 1, 5, 50.
- Output bounds verified: steering in [-1, 1] (tanh), acceleration and
  eat_signal in [0, 1] (sigmoid).

## Inputs to Phase 5 Compute Budget

- Per-step batched inference cost at 250 agents: ~0.093 ms (forward pass only,
  pre-stacked weights).
- Weight stacking is the amortizable cost; the benchmark excludes stacking
  (stacked once per generation in the production flow per PLAN.md).
- Headroom of 358x means Phase 2's sensor computation and rendering can each
  consume substantial budget before inference becomes the bottleneck.

## Notes

- First-report numbers from a CPU-only reference machine; re-run the
  benchmark before quoting performance-sensitive claims on other hardware.