# Roblox Up-and-Coming top-10 forecaster

Entry for the make.fun bounty "Predict Roblox's Up-and-Coming top 10"
(`roblox-up-and-coming-forecast-01`, $500). A rule-based forecaster in
Python 3.13, stdlib only, no network, no model calls.

## What it does

`main.py` reads `/input/history.ndjson` (one JSON object per read of the
Up-and-Coming chart, oldest first) and `/input/task.json`, and writes
`/output/forecast.json`: per-place probabilities for the chart's top 10
plus a `top10` guess, scored over the 72 hours after the history ends.
It honours `$MAKEFUN_INPUT_DIR` / `$MAKEFUN_OUTPUT_DIR` when set.

## Model

The chart is volatile — the top 10 changed in 42 of 44 consecutive read
pairs in the archive, a median of 6 places — and it sorts by user growth,
so a no-change forecast decays fast (the house baseline scores -43.51 on
the practice window). This entry forecasts by forward simulation:

1. **Features.** For every game on the final chart: current rank, log
   playing count, 24h and 48h log playing growth, recent rank velocity,
   and hours since it first appeared on the chart. When the history
   carries games/v1 `details`, each game's `created` date is read too, so
   a game whose 28-day chart life ends inside the window is removed from
   the simulation from that hour on.
2. **Rank regressions, fit from the history itself.** One linear regression
   per horizon bucket (0–12h, 12–24h, 24–36h, 36–60h, 60–84h): the rank a
   game will hold that far ahead, given its features now. Recent playing
   growth is the strongest predictor of a rank gain (correlation 0.27 with
   the next day's rank gain); current rank fades with horizon; age on the
   chart drags a game down. The fit is pure Python (normal equations,
   Gaussian elimination), so the entry adapts to whatever history it is
   given — the final history is longer and hourly, unlike the practice
   one.
3. **Rank-dependent spread and bias.** The top of the chart is far more
   predictable than its tail (residual std ≈ 3 ranks at 0–12h for the top
   10, ≈ 10 for rank 61+), and the regression systematically underrates
   top games (their mean residual is negative and grows with horizon), so
   both the spread and the mean are corrected per current-rank band.
4. **Monte-Carlo simulation.** The chart is simulated forward, hour by
   hour, for 72 hours, 300 seeded repetitions: each game's rank at hour
   `t` is drawn from a Gaussian centred on the regression's prediction
   (coefficients and spread interpolated across the horizon buckets,
   anchored at `t=0` to the game's exact current rank), the games are
   sorted into a chart, and the places are tallied. One fixed seed, so two
   runs on the same history are byte-identical.
5. **Probabilities.** The tallies become the per-place probabilities. Each
   place leaves 10% unnamed for games not on the chart yet — new entrants
   do reach the top 10 (three did in the practice window, one of them #1)
   — and for model error. The `top10` is the ten games most likely to be
   in the top 10 at all, ordered by best expected rank.

## Verification

- The practice window's 10 reads were reconstructed from the public
  archive (github.com/Kenny-Dhub/roblox-snapshots, `data/discover/latest.json`
  at the 10 commits in the window) and scored with the rules' log-score
  formula. The scorer was validated first: it reproduces the house
  baseline's published practice score exactly (-43.512039).
- This entry scores **-33.72** on the practice window (typical 0.0343 per
  place), vs the house baseline's -43.51 (typical 0.0129) — about 9.8
  log points better, 2.7× the typical probability. Top-10 hits 6.3/read
  (baseline 6.4; the log score is what ranks).
- Rolling backtests inside the history (forecast from reads 0–19, 0–23,
  0–27; score on the held-out reads) also beat the baseline at every cut
  (-44.2 / -42.0 / -30.0 vs -53.0 / -56.9 / -37.7). The chosen parameters
  (rank-band spread + bias, sigma scale 1.0, 10% unnamed) were the ones
  that won both the true window and the rolling backtests; shrinking the
  spread further won the true window but lost the rolling ones, so it was
  not kept.
- Two runs on the kit input produce byte-identical `forecast.json`.
  Output is valid: exactly two keys, per-place sums ≤ 1 (0.89 named +
  0.10 unnamed), at most 100 rows per place, `top10` exactly 10 distinct
  ids, 43,811 bytes (limit 1,048,576). Runtime ~17 s (limit 300 s).
  Stdlib only: `json`, `math`, `os`, `random`, `datetime`.

## Limits

- The model only knows games already on the chart: a brand-new game that
  rockets into the top 10 can only be covered by the unnamed share.
- The rank regressions are linear; they cannot express interactions (e.g.
  a growth spike mattering more for a young game than an old one).
- `visits`/`favorites` growth from `details` is not used as a feature
  (the practice history has no details to fit it on); it is only the
  `created` date that feeds the aging-out rule.
- The practice window is 10 reads, so the headline number is a small
  sample; the rolling backtests are the more robust evidence.
