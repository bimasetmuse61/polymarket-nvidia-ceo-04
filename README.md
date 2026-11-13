# polymarket-nvidia-ceo-04

Fourth entry for **polymarket-nvidia-ceo-forecast-01** ($250): forecast
Polymarket's NVIDIA CEO market prices, minute by minute, four hours
ahead, as five quantiles (q05/q25/q50/q75/q95) per minute.

## Design

A two-space empirical ensemble, refined from the third entry:

- For each rung of a horizon ladder (1, 2, 4, 8, 15, 30, 60, 120, 240
  minutes), collect that horizon's historical price changes and take
  their quantiles. A market's own quantiles (recency-weighted) are
  blended with a pooled shape built from every market in the case,
  each normalised by its own typical step size (the 70th percentile
  of its absolute changes at that horizon).
- The whole construction runs twice: once in price space (half-life
  120 min, pool share 0.50) and once in logit space (half-life
  150 min, pool share 0.55), the latter mapped back through the
  sigmoid.
- Spreads are contracted toward the median (inner quantiles 0.65 in
  both views, outer 0.30) and scaled by a volatility-regime factor:
  the last day's mean absolute one-minute step over the full
  window's, clipped to [0.5, 2.0]. The median itself never moves —
  neither drift nor mean-reversion survived the practice data.

**What changed in v4:** the ensemble weight is no longer flat. Near
the 0/1 boundaries, logit-space changes are mechanically inflated (a
one-tick move at p = 0.01 is an enormous logit step), and the logit
view proves the weaker witness there. The logit weight therefore
ramps with the checkpoint price's distance from the nearer boundary:
0.72 once the price is at least 0.05 from both edges, easing linearly
to a 0.40 floor at the edges. Inner-quantile contraction was also
relaxed from 0.55/0.60 to 0.65 in both views.

## Validation (local practice scorer, rebuilt 1-minute grid)

| entry | practice loss |
| ----- | ------------- |
| house baseline | 0.363589 |
| v1 | 0.303524 |
| v2 | 0.285430 |
| v3 | 0.269371 |
| **v4** | **0.268883** |

Split-half check against v3: H1 (p00–p02) 0.374091 → 0.373144,
H2 (p03–p05) 0.185300 → 0.185181 — both halves improve. Deterministic:
two runs over all six practice cases produce byte-identical output.
Heaviest practice case (24 markets) runs in ~1.4 s.

## Running

The harness supplies `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`;
defaults are `/input` and `/output`. The program reads every `*.in`
case file and writes one `<case>.out` per case. Python 3.13,
standard library only, no network.
