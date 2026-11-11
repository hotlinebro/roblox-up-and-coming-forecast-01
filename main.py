#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (rule-based, no model calls).

Reads /input/history.ndjson (one JSON object per read of the chart, oldest
first) and /input/task.json, and writes /output/forecast.json: per-place
probabilities for the chart's top 10 plus a top10 guess, scored over the
72 hours after the history ends.

Model
-----
The chart is volatile (median 6 of the top 10 places change between reads)
and sorts by user growth, so a no-change forecast decays fast. This entry
forecasts by forward simulation:

1. For every game on the final chart it builds features from its history:
   current rank, log playing count, 24h and 48h log playing growth, recent
   rank velocity, and hours since it first appeared on the chart. When the
   history carries games/v1 details, it also reads each game's `created`
   date, so a game whose 28-day chart life ends inside the window is
   removed from the simulation from that hour on.
2. It fits, from the history itself, one linear rank regression per
   horizon bucket (0-12h, 12-24h, 24-36h, 36-60h, 60-84h): the rank a game
   will hold that far ahead, given its features now. Recent playing growth
   is the strongest predictor of a rank gain; current rank fades with
   horizon; age on the chart drags a game down. The fit is done in pure
   Python (normal equations, Gaussian elimination), so the entry adapts to
   whatever history it is given.
3. It Monte-Carlo simulates the chart forward, hour by hour, for 72 hours:
   each game's rank at hour t is drawn from a Gaussian centred on the
   regression's prediction (coefficients and residual spread interpolated
   across the horizon buckets, anchored at t=0 to the game's exact current
   rank), the games are sorted into a chart, and the places are tallied.
   The draws use one seeded RNG, so two runs on the same history are
   byte-identical.
4. The tallies become the per-place probabilities. Each place leaves a
   share unnamed for games that are not on the chart yet (new entrants do
   reach the top 10: three did in the practice window, one of them #1), and
   for model error. The top10 is the ten games with the best expected rank.

Stdlib only. No network, no model calls, no wall clock in the forecast
logic: every time used comes from the history's own timestamps.
"""

import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Horizon buckets (hours) for the rank regressions, and their centres.
BUCKETS = [(0.0, 12.0), (12.0, 24.0), (24.0, 36.0), (36.0, 60.0), (60.0, 84.0)]
CENTRES = [6.0, 18.0, 30.0, 48.0, 72.0]
CHART_LIFE_HOURS = 28.0 * 24.0  # games leave the chart 28 days after publication

N_SIMS = 300        # Monte-Carlo repetitions of the 72-hour window
UNNAMED = 0.10      # share of each place left unnamed (new entrants + model error)
SIGMA_SCALE = 1.0   # multiplier on the regression residual spread
SEED = 20261111     # fixed seed: two runs on the same history are identical
MAX_NAMED_PER_PLACE = 100
# These three were tuned on the practice window (true 10-read window) and
# on rolling backtests inside the history; the values above are the ones
# that won both: rank-band spread + bias correction, sigma scale 1.0,
# 10% unnamed. (Practice window: -33.72 vs the house baseline's -43.51.)
# Rank bands for the residual spread / bias: the top of the chart is far
# more predictable than its tail (residual std ~3 ranks at 0-12h for the
# top 10, ~10 for rank 61+), and the regression systematically underrates
# top games (their mean residual is negative and grows with horizon).
BANDS = [(1, 10), (11, 30), (31, 60), (61, 10 ** 9)]
BAND_CENTRES = [5.0, 20.0, 45.0, 80.0]



def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_history(input_dir):
    recs = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: parse_time(r["at"]))
    task = json.load(open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8"))
    return recs, task


def solve_linear(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            m[piv][col] = 1e-12
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / pv
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def fit_regression(xs, ys):
    """Least squares y ~ X (xs already include a constant column).

    Returns (coefficients, residual_std). Falls back to a persistence-only
    model when there are too few samples.
    """
    n_feat = len(xs[0])
    if len(xs) < 5 * n_feat:
        return None, None
    # Normal equations: (X'X) b = X'y
    ata = [[0.0] * n_feat for _ in range(n_feat)]
    aty = [0.0] * n_feat
    for x, y in zip(xs, ys):
        for i in range(n_feat):
            aty[i] += x[i] * y
            for j in range(n_feat):
                ata[i][j] += x[i] * x[j]
    # Tiny ridge for stability.
    for i in range(n_feat):
        ata[i][i] += 1e-6
    coef = solve_linear(ata, aty)
    ss = 0.0
    for x, y in zip(xs, ys):
        pred = sum(c * v for c, v in zip(coef, x))
        ss += (y - pred) ** 2
    dof = max(len(xs) - n_feat, 1)
    return coef, math.sqrt(ss / dof)


def point_before(series, t_target):
    """Latest (t, rank, playing) in a game's series at or before t_target."""
    best = None
    for pt in series:
        if pt[0] <= t_target and (best is None or pt[0] > best[0]):
            best = pt
    return best


def game_features(series, t0, rank0, play0):
    """Feature vector for one game at one read.

    [1, rank, log playing, 24h log growth, 48h log growth,
     rank velocity (ranks/hour gained, + = climbing), age on chart (h)]
    """
    p24 = point_before(series, t0 - 24.0 * 3600.0)
    p48 = point_before(series, t0 - 48.0 * 3600.0)
    first_t = series[0][0]
    age_h = (t0 - first_t) / 3600.0
    g24 = math.log(play0 / p24[2]) if p24 and p24[2] and play0 else 0.0
    g48 = math.log(play0 / p48[2]) if p48 and p48[2] and play0 else 0.0
    vel24 = (p24[1] - rank0) / max((t0 - p24[0]) / 3600.0, 1.0) if p24 else 0.0
    return [1.0, float(rank0), math.log(max(play0, 1)), g24, g48, vel24, age_h]


def interp(anchors, t):
    """Linear interpolation of a per-bucket value across horizon centres.

    anchors: list of values at CENTRES. At t=0 returns the first anchor
    scaled toward the identity; the caller handles the t=0 anchor itself.
    """
    if t <= CENTRES[0]:
        return anchors[0]
    if t >= CENTRES[-1]:
        return anchors[-1]
    for i in range(len(CENTRES) - 1):
        if CENTRES[i] <= t <= CENTRES[i + 1]:
            f = (t - CENTRES[i]) / (CENTRES[i + 1] - CENTRES[i])
            return anchors[i] * (1.0 - f) + anchors[i + 1] * f
    return anchors[-1]


def interp_centres(centres, values, x):
    """Linear interpolation of a value across arbitrary centres."""
    if x <= centres[0]:
        return values[0]
    if x >= centres[-1]:
        return values[-1]
    for i in range(len(centres) - 1):
        if centres[i] <= x <= centres[i + 1]:
            f = (x - centres[i]) / (centres[i + 1] - centres[i])
            return values[i] * (1.0 - f) + values[i + 1] * f
    return values[-1]


def main():
    recs, task = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    window = task.get("window", {})
    n_hours = int(window.get("slots", 72))
    t_end = parse_time(recs[-1]["at"])

    # Per-game series: uid -> [(t, rank, playing, name)], and created dates.
    series = {}
    created = {}
    for r in recs:
        t = parse_time(r["at"])
        for g in r.get("up_and_coming") or []:
            series.setdefault(g["universe_id"], []).append(
                (t, g["rank"], g.get("playing") or 0, g.get("name") or "")
            )
        for d in r.get("details") or []:
            uid = d.get("universe_id")
            if uid is not None and d.get("created"):
                try:
                    created[uid] = parse_time(d["created"])
                except (ValueError, TypeError):
                    pass
    for s in series.values():
        s.sort(key=lambda p: p[0])

    # ---- Fit one rank regression per horizon bucket, from the history. ----
    buckets_x = [[] for _ in BUCKETS]
    buckets_y = [[] for _ in BUCKETS]
    for i, r in enumerate(recs):
        t0 = parse_time(r["at"])
        for g in r.get("up_and_coming") or []:
            uid = g["universe_id"]
            s = series[uid]
            feats = game_features(s, t0, g["rank"], g.get("playing") or 0)
            for j in range(i + 1, len(recs)):
                t1 = parse_time(recs[j]["at"])
                dt = (t1 - t0) / 3600.0
                b = None
                for bi, (lo, hi) in enumerate(BUCKETS):
                    if lo <= dt < hi:
                        b = bi
                        break
                if b is None:
                    continue
                fmap = {x["universe_id"]: x for x in recs[j].get("up_and_coming") or []}
                if uid in fmap:
                    buckets_x[b].append(feats)
                    buckets_y[b].append(float(fmap[uid]["rank"]))
    coefs = []
    sigmas = []
    band_sigmas = []   # per bucket: residual std per rank band
    band_means = []    # per bucket: mean residual per rank band (bias)
    for b in range(len(BUCKETS)):
        xs, ys = buckets_x[b], buckets_y[b]
        coef, sig = fit_regression(xs, ys)
        if coef is None:
            # Fallback: persistence (rank stays), with a wide spread.
            coef = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            sig = 20.0
        coefs.append(coef)
        sigmas.append(sig)
        # Residuals per current-rank band (rank0 is feature index 1).
        bs, bm = [], []
        for (rlo, rhi) in BANDS:
            res = []
            for x, y in zip(xs, ys):
                if rlo <= x[1] <= rhi:
                    pred = sum(c * v for c, v in zip(coef, x))
                    res.append(y - pred)
            if len(res) >= 10:
                m = sum(res) / len(res)
                var = sum((v - m) ** 2 for v in res) / max(len(res) - 1, 1)
                bs.append(math.sqrt(var))
                bm.append(m)
            else:
                bs.append(sig)
                bm.append(0.0)
        band_sigmas.append(bs)
        band_means.append(bm)

    # ---- Games on the final chart, with features and exit times. ----
    last = recs[-1]
    final = sorted(last.get("up_and_coming") or [], key=lambda g: g["rank"])
    games = []
    for g in final:
        uid = g["universe_id"]
        s = series[uid]
        feats = game_features(s, t_end, g["rank"], g.get("playing") or 0)
        exit_t = None
        if uid in created:
            exit_t = created[uid] + CHART_LIFE_HOURS
        games.append({
            "uid": uid,
            "rank0": float(g["rank"]),
            "feats": feats,
            "exit_t": exit_t,
        })
    if not games:
        raise SystemExit("no games on the final chart")

    # ---- Monte-Carlo simulation of the next n_hours hours. ----
    n_sims = N_SIMS
    unnamed = UNNAMED
    sigma_scale = SIGMA_SCALE
    use_band = 1
    use_bias = 1
    rng = random.Random(SEED)
    tally = {}          # (uid, place) -> count
    expected_rank = {}  # uid -> mean simulated rank (for the top10)
    n_draws = n_sims * n_hours
    for _sim in range(n_sims):
        for h in range(1, n_hours + 1):
            t = t_end + h * 3600.0
            # Interpolated coefficients / spread for this horizon.
            coef_t = [interp([c[k] for c in coefs], float(h)) for k in range(len(coefs[0]))]
            # Blend toward the exact current rank at short horizons.
            w = min(h / CENTRES[0], 1.0)
            draws = []
            for gm in games:
                if gm["exit_t"] is not None and t >= gm["exit_t"]:
                    continue
                mean = sum(c * v for c, v in zip(coef_t, gm["feats"]))
                if use_bias:
                    bias_t = interp(
                        [interp_centres(BAND_CENTRES, band_means[b], gm["rank0"])
                         for b in range(len(BUCKETS))],
                        float(h),
                    )
                    mean += bias_t
                mean = w * mean + (1.0 - w) * gm["rank0"]
                if use_band:
                    sig_t = interp(
                        [interp_centres(BAND_CENTRES, band_sigmas[b], gm["rank0"])
                         for b in range(len(BUCKETS))],
                        float(h),
                    )
                else:
                    sig_t = interp(sigmas, float(h))
                sig = w * sig_t * sigma_scale
                r = rng.gauss(mean, sig) if sig > 0 else mean
                draws.append((r, gm["uid"]))
            draws.sort(key=lambda d: d[0])
            for place, (_r, uid) in enumerate(draws[:places], start=1):
                tally[(uid, place)] = tally.get((uid, place), 0) + 1
            for _r, uid in draws:
                expected_rank[uid] = expected_rank.get(uid, 0.0) + _r
    for uid in expected_rank:
        expected_rank[uid] /= n_draws

    # ---- Probabilities: tallies scaled to leave UNNAMED unnamed. ----
    probabilities = []
    for place in range(1, places + 1):
        rows = sorted(
            ((uid, c) for (uid, pl), c in tally.items() if pl == place),
            key=lambda uc: -uc[1],
        )[:MAX_NAMED_PER_PLACE]
        total = sum(c for _, c in rows)
        if total <= 0:
            continue
        for uid, c in rows:
            p = (1.0 - unnamed) * c / total
            if p > 0.0005:
                probabilities.append({"universe_id": uid, "place": place, "p": round(p, 6)})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # Top 10: the games most likely to be in the top 10 at all (sum of their
    # per-place probabilities), ordered by best expected rank.
    in_top10 = {}
    for (uid, _pl), c in tally.items():
        in_top10[uid] = in_top10.get(uid, 0) + c
    top10 = sorted(in_top10, key=lambda u: (-in_top10[u], expected_rank.get(u, 1e9)))[:places]
    # Fill any shortfall (should not happen) with the final chart's order.
    if len(top10) < places:
        seen = set(top10)
        for g in final:
            if g["uid"] not in seen:
                top10.append(g["uid"])
            if len(top10) >= places:
                break

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
