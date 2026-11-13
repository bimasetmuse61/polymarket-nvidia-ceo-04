#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01, fourth entry: a two-space ensemble
whose trust in the logit view fades near the price boundaries.

The starting point is the behaviour of these markets: prices sit near
zero for days, moving in small ticks with the odd jump. For each rung
of a horizon ladder (1..240 minutes) the forecaster gathers that
horizon's historical price changes and reads off their quantiles; the
checkpoint price is then spread by those quantiles, contracted toward
the median and rescaled by how lively the last day was relative to
the whole window (the regime factor, clipped to [0.5, 2.0]).

Two views of the same history are maintained:

* Price space. A market's own recency-weighted change quantiles
  (exponential decay, half-life 120 min) are blended half-and-half
  with a pooled shape: the change distributions of every market in
  the case, each normalised by its own typical step (the 70th
  percentile of its absolute changes at that horizon, robust to rare
  jumps), merged and rescaled back to the market at hand.
* Logit space. The identical construction on logit(p), with a
  slightly longer memory (half-life 150 min) and slightly stronger
  pooling (0.55). The sigmoid carries the quantiles home.

What is new in this entry is how the two views are mixed. In the
third entry the logit view carried a flat 0.70 weight everywhere.
Near the 0/1 boundaries, though, logit changes are mechanically
inflated -- a one-tick step at p = 0.01 is a huge logit step -- and
the practice cases say the logit view is the less reliable witness
there. So the logit weight now ramps with the checkpoint price's
distance from the nearer boundary: full 0.72 when the price is at
least 0.05 away from both edges, easing down to a 0.40 floor exactly
at the edges, linear in between. Interior quantiles are also
contracted a little less than before (0.65 in both views instead of
0.55/0.60); the outer quantiles stay at 0.30, because the far tails
of an empirical change distribution overstate a single fresh window.

The median never moves: drift and mean-reversion were tried in
earlier entries and the data refused them. Rung quantiles are
interpolated in sqrt(h); every row is rounded to 6 decimals and
forced non-decreasing.

Python 3.13, standard library only, no network, deterministic.
Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input) and writes
one <case>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
LADDER = (1, 2, 4, 8, 15, 30, 60, 120, 240)
STEP_LEVEL = 0.70      # percentile of |change| defining a market's step size
REGIME_DAYS = 1440     # minutes in the regime factor's recent window
REGIME_LO = 0.5
REGIME_HI = 2.0
CLAMP = 1e-6           # logit clamp distance from the boundaries
LN2 = math.log(2.0)

# View parameters: half-life, pool share, inner contraction, outer contraction.
PRICE_VIEW = (120.0, 0.50, 0.65, 0.30)
LOGIT_VIEW = (150.0, 0.55, 0.65, 0.30)
# Logit blend weight: FLOOR at the boundary, TOP once the checkpoint
# price is at least REACH away from both boundaries.
W_FLOOR = 0.40
W_TOP = 0.72
W_REACH = 0.05


def plain_quantile(ascending, tau):
    """Type-7 (linear interpolation) quantile of an ascending sequence."""
    n = len(ascending)
    if n == 1:
        return ascending[0]
    pos = tau * (n - 1)
    lo = int(pos)
    if lo >= n - 1:
        return ascending[-1]
    frac = pos - lo
    return ascending[lo] * (1.0 - frac) + ascending[lo + 1] * frac


def weighted_quantiles(pairs):
    """The five TAUS quantiles of (value, weight) pairs sorted by value.

    Each sample sits at the midpoint of its own slice of the total
    weight and the levels are interpolated between those midpoints.
    """
    total = sum(w for _, w in pairs)
    marks = []
    run = 0.0
    for _, w in pairs:
        run += w
        marks.append((run - 0.5 * w) / total)
    out = []
    j = 0
    for tau in TAUS:
        if tau <= marks[0]:
            out.append(pairs[0][0])
            continue
        if tau >= marks[-1]:
            out.append(pairs[-1][0])
            continue
        while marks[j + 1] < tau:
            j += 1
        span = marks[j + 1] - marks[j]
        f = 0.0 if span == 0.0 else (tau - marks[j]) / span
        out.append(pairs[j][0] * (1.0 - f) + pairs[j + 1][0] * f)
    return out


def changes_at(series, h):
    return [series[i + h] - series[i] for i in range(len(series) - h)]


def own_quantiles(series, h, changes, half_life):
    """Recency-weighted quantiles of one market's h-minute changes.

    A change's age is the distance from the minute it ends to the end
    of the series, so recent behaviour speaks loudest.
    """
    n = len(series)
    pairs = [(ch, math.exp(-((n - h) - i) * LN2 / half_life))
             for i, ch in enumerate(changes)]
    pairs.sort(key=lambda p: p[0])
    return weighted_quantiles(pairs)


def regime(series):
    """Last day's mean |1-minute step| over the full window's, clipped."""
    steps = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
    if len(steps) < REGIME_DAYS:
        return 1.0
    full = sum(steps) / len(steps)
    if full <= 0.0:
        return 1.0
    recent = sum(steps[-REGIME_DAYS:]) / REGIME_DAYS
    return min(REGIME_HI, max(REGIME_LO, recent / full))


def rung_row(table, h):
    """Quantile row at horizon h, interpolated in sqrt(h) over the ladder."""
    hs = sorted(table)
    if h <= hs[0]:
        return table[hs[0]]
    if h >= hs[-1]:
        return table[hs[-1]]
    for a, b in zip(hs, hs[1:]):
        if a <= h <= b:
            f = (math.sqrt(h) - math.sqrt(a)) / (math.sqrt(b) - math.sqrt(a))
            return [table[a][k] * (1.0 - f) + table[b][k] * f for k in range(5)]
    return table[hs[-1]]


def view_grids(series_list, half_life, pool_share, inner, outer, horizon,
               clip=False):
    """One view's 240 x 5 quantile grid per market, in the view's space.

    First pass: per market and rung, the changes and the typical step
    size; the pooled, size-normalised changes shared by the case.
    Second pass: own quantiles blended with the pooled shape, spread
    around the last value, contracted, regime-scaled.
    """
    prepared = []
    pool_raw = {h: [] for h in LADDER}
    for series in series_list:
        per_rung = {}
        for h in LADDER:
            if h >= len(series):
                continue
            ch = changes_at(series, h)
            size = plain_quantile(sorted(abs(c) for c in ch), STEP_LEVEL) if ch else 0.0
            per_rung[h] = (ch, size)
            if size > 0.0:
                pool_raw[h].extend(c / size for c in ch)
        prepared.append((series, per_rung))
    pool_shape = {}
    for h, vals in pool_raw.items():
        if vals:
            vals.sort()
            pool_shape[h] = [plain_quantile(vals, t) for t in TAUS]

    scales = (outer, inner, 1.0, inner, outer)
    grids = []
    for series, per_rung in prepared:
        centre = series[-1] if series else 0.0
        table = {}
        for h, (ch, size) in per_rung.items():
            row = own_quantiles(series, h, ch, half_life)
            if h in pool_shape and size > 0.0:
                row = [(1.0 - pool_share) * row[k]
                       + pool_share * pool_shape[h][k] * size
                       for k in range(5)]
            table[h] = row
        factor = regime(series)
        grid = []
        for h in range(1, horizon + 1):
            if table:
                qs = rung_row(table, h)
                med = qs[2]
                row = [centre + (qs[k] - med) * scales[k] * factor
                       for k in range(5)]
            else:
                row = [centre] * 5
            if clip:
                row = [min(1.0, max(0.0, v)) for v in row]
            for k in range(1, 5):
                if row[k] < row[k - 1]:
                    row[k] = row[k - 1]
            grid.append(row)
        grids.append(grid)
    return grids


def sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def round6(v):
    r = round(min(1.0, max(0.0, v)), 6)
    return r + 0.0  # fold -0.0 into 0.0


def to_logit(p):
    p = min(1.0 - CLAMP, max(CLAMP, p))
    return math.log(p / (1.0 - p))


def logit_weight(last_price):
    """Trust in the logit view, ramped by distance from the boundary."""
    dist = min(last_price, 1.0 - last_price)
    f = min(1.0, dist / W_REACH)
    return W_FLOOR + (W_TOP - W_FLOOR) * f


def forecast_case(markets, horizon):
    price_series = []
    logit_series = []
    for m in markets:
        prices = [float(p) for p in m["prices"]]
        price_series.append(prices)
        logit_series.append([to_logit(p) for p in prices])

    phl, ppool, pin, pout = PRICE_VIEW
    lhl, lpool, lin, lout = LOGIT_VIEW
    price_grids = view_grids(price_series, phl, ppool, pin, pout, horizon,
                             clip=True)
    logit_grids = view_grids(logit_series, lhl, lpool, lin, lout, horizon)

    forecasts = []
    for idx, m in enumerate(markets):
        prices = price_series[idx]
        if len(prices) < 3:
            flat = round6(prices[-1]) if prices else 0.5
            forecasts.append({"id": m["id"],
                              "q": [[flat] * 5 for _ in range(horizon)]})
            continue
        w = logit_weight(prices[-1])
        rows = []
        for h in range(horizon):
            row = []
            for k in range(5):
                v = ((1.0 - w) * price_grids[idx][h][k]
                     + w * sigmoid(logit_grids[idx][h][k]))
                row.append(round6(v))
            for k in range(1, 5):
                if row[k] < row[k - 1]:
                    row[k] = row[k - 1]
            rows.append(row)
        forecasts.append({"id": m["id"], "q": rows})
    return forecasts


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(f for f in os.listdir(input_dir) if f.endswith(".in")):
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecasts = forecast_case(case["markets"], int(case["horizon_minutes"]))
        out_path = os.path.join(output_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"forecasts": forecasts}, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
