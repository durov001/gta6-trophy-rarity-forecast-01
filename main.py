"""Forecast GTA VI trophy earn rates on Exophase at the case target.

Method (own implementation, stdlib only):
  - Rate = E / N, modelled separately. E is the earners of a trophy, N is
    players_tracked. New players both dilute the rate and add earners, so
    the two are never forecast independently.
  - Launch-window slopes are contaminated: the first 48 hours are the
    explosive-N phase, the 4.5-day horizon is the slowing phase. So every
    fitted slope is shrunk (empirical Bayes, x0.3), fits use only the last
    18 hours with exponential weights (12 h half-life), and the N lever arm
    is damped (saturation: N growth decays toward zero over the horizon).
  - Paired scenarios, not averaged quantiles: 3 N-growth scenarios x 3
    progression speeds = 9 paths. Each N path carries a consistent E path:
    new players earn at a factor f of the existing rate (f near 1 for
    common trophies, ~0.35 for rare ones), so a bigger N jump also lifts E.
    Persistence is a 10th path (an anchor, not the centre). The centre is
    the median of the 10 paths on the logit scale.
  - Spread is a variance sum, not a practice fit: (lever arm x slope SE)^2
    + refresh variance (measured from the logit jumps when N surged in the
    history) + count variance (1/sqrt(E+1)) + a floor. Quantiles use t(4):
    median -/+ 2.13 sigma at 5/95%, -/+ 0.74 sigma at 25/75%. Late
    story/collection trophies (0.1-5%) get an asymmetric band (upper tail
    wider: +2.0 sigma up, -1.2 sigma down); platinum gets a wide floor.
  - Clamp: every quantile inside [0.85x, 1.5x] the trophy's last rate
    (platinum up to 3x); early-story trophies cap q95 at 95%.

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes one .out per
case to $MAKEFUN_OUTPUT_DIR (else /output). Deterministic: no randomness,
fixed grids and iteration counts, files processed in name order.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
# t(4) multipliers for the 5/95 and 25/75 quantiles.
T4_OUTER = 2.13
T4_INNER = 0.74
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

SLOPE_SHRINK = 0.3        # empirical-Bayes shrink on every fitted slope
FIT_WINDOW_H = 18.0       # fit only the last 18 hours of reads
HALF_LIFE_H = 12.0        # exponential recency weights
SAT_SCALE = 1.5           # saturation scale for the N lever arm (log-time)
N_SCEN = (0.5, 1.0, 2.0)  # N-growth scenario multipliers
P_SCEN = (0.5, 1.0, 2.0)  # progression scenario multipliers


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves up."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def median_of(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def std_of(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return math.sqrt(sum((v - m) ** 2 for v in vals) / n)


def wls_stats(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b, xbar_w).

    se_b is the standard error of the slope from the weighted residual
    variance; 0 when the fit is exact or degenerate.
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        return (my, 0.0, 0.0, mx)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    s2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(n - 2, 1)
    se_b = math.sqrt(max(s2, 0.0) / dof / sxx) if s2 > 0 else 0.0
    return (a, b, se_b, mx)


def exp_weight(age_hours):
    return 0.5 ** (age_hours / HALF_LIFE_H)


def refresh_stats(history, ns):
    """Typical logit shock per N refresh and the gap between refreshes.

    A refresh is a read where N jumps >2% over the previous read. The shock
    is the median absolute logit move of the trophies across the jump.
    Falls back to priors when the history shows no jump.
    """
    shocks = []
    jump_times = []
    for i in range(1, len(history)):
        if ns[i - 1] <= 0 or ns[i] / ns[i - 1] <= 1.02:
            continue
        moves = []
        before = history[i - 1]["rates"]
        after = history[i]["rates"]
        for rb, ra in zip(before, after):
            if rb is None or ra is None:
                continue
            moves.append(abs(logit(float(ra)) - logit(float(rb))))
        if len(moves) >= 5:
            shocks.append(median_of(moves))
            jump_times.append(parse_time(history[i]["at"]))
    if not shocks:
        return 0.3, 6.0
    gaps = [(jump_times[k + 1] - jump_times[k]).total_seconds() / 3600.0
            for k in range(len(jump_times) - 1) if jump_times[k + 1] > jump_times[k]]
    gap = median_of(gaps) if gaps else 6.0
    if gap <= 0:
        gap = 6.0
    return median_of(shocks), gap


def forecast_case(case):
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target = parse_time(case["target"])
    t_target_h = (target - RELEASE).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target_h, 0.0))

    history = case["history"]
    n_reads = len(history)
    t_last = parse_time(history[-1]["at"])

    xs, ns, times = [], [], []
    last_n = 1.0
    for read in history:
        at = parse_time(read["at"])
        times.append(at)
        t_h = (at - RELEASE).total_seconds() / 3600.0
        xs.append(math.log1p(max(t_h, 0.0)))
        n = read.get("players_tracked")
        if n is None or n <= 0:
            n = last_n
        ns.append(float(n))
        last_n = float(n)
    last_n = ns[-1]
    x_last = xs[-1]
    log_n_last = math.log(last_n)

    # Fit window: last 18 h (or everything when the history is shorter).
    win = [i for i in range(n_reads)
           if (t_last - times[i]).total_seconds() <= FIT_WINDOW_H * 3600.0]
    if len(win) < 3:
        win = list(range(n_reads))
    w_win = [exp_weight((t_last - times[i]).total_seconds() / 3600.0) for i in win]

    # --- N: log(N) on log-time, slope shrunk, lever arm saturated.
    _, s_n, _, _ = wls_stats([xs[i] for i in win], [math.log(ns[i]) for i in win], w_win)
    arm = max(x_target - x_last, 0.0)
    damped_arm = SAT_SCALE * (1.0 - math.exp(-arm / SAT_SCALE)) if arm > 0 else 0.0
    g_n = SLOPE_SHRINK * s_n * damped_arm
    if g_n < 0.0:
        g_n = 0.0
    n_mid = math.exp(log_n_last + g_n)
    n_paths = [math.exp(log_n_last + m * g_n) for m in N_SCEN]

    # --- Refresh variance from the history's own N jumps.
    shock, gap_h = refresh_stats(history, ns)
    n_refresh = horizon_hours / gap_h if gap_h > 0 else 0.0
    sig_refresh = shock * math.sqrt(min(n_refresh, 4.0))

    rows = []
    for ti, trophy in enumerate(case["trophies"]):
        last_rate = None
        for read in reversed(history):
            r = read["rates"][ti]
            if r is not None:
                last_rate = float(r)
                break
        if last_rate is None:
            last_rate = 0.0
        e_last = last_rate * last_n / 100.0

        # E fit on the window: log(E + 0.5) on log-time.
        ex, ey, ew = [], [], []
        for k, i in enumerate(win):
            r = history[i]["rates"][ti]
            if r is None:
                continue
            ex.append(xs[i])
            ey.append(math.log(float(r) * ns[i] / 100.0 + 0.5))
            ew.append(w_win[k])
        if len(ex) < 2:
            ex, ey, ew = xs, [math.log(e_last + 0.5)] * n_reads, [1.0] * n_reads
        _, s_e, se_e, xbar = wls_stats(ex, ey, ew)
        g_e = SLOPE_SHRINK * s_e * arm
        g_e = min(max(g_e, -0.2), 2.0)

        # New-player earn factor: common trophies are earned by newcomers
        # almost as fast as by incumbents; rare ones barely at all.
        f_new = 0.3 + 0.7 * math.sqrt(clamp_rate(last_rate) / 100.0)

        # 9 paired paths + persistence as the 10th.
        lg_paths = []
        for n_s in n_paths:
            for m_p in P_SCEN:
                e_pair = (e_last + 0.5) * math.exp(m_p * g_e) * ((n_s / n_mid) ** f_new) - 0.5
                if e_pair < 0.0:
                    e_pair = 0.0
                lg_paths.append(logit(100.0 * e_pair / n_s))
        lg_paths.append(logit(last_rate))
        centre = median_of(lg_paths)

        # Spread: scenario dispersion + slope uncertainty + refresh + count + floor.
        sig_paths = std_of(lg_paths)
        sig_slope = max(x_target - xbar, 0.0) * se_e
        sig_count = 1.0 / math.sqrt(e_last + 1.0)
        if trophy.get("tier") == "platinum":
            sig_floor = 0.4
        elif last_rate < 1.0:
            sig_floor = 0.3
        else:
            sig_floor = 0.15
        sigma = math.sqrt(sig_paths ** 2 + sig_slope ** 2 + sig_refresh ** 2
                          + sig_count ** 2 + sig_floor ** 2)
        sigma = min(sigma, 1.5)

        # Quantiles on the logit scale; late story/collection trophies get
        # an asymmetric band (upper tail wider).
        if 0.1 <= last_rate <= 5.0:
            offs = (-1.2, -T4_INNER, 0.0, T4_INNER, 2.0)
        else:
            offs = (-T4_OUTER, -T4_INNER, 0.0, T4_INNER, T4_OUTER)
        qs = [inv_logit(centre + o * sigma) for o in offs]

        # Early-story trophies near saturation: cap the top quantile.
        if last_rate >= 40.0:
            qs[4] = min(qs[4], 95.0)

        # Near-certain trophies: the logit scale cannot express a dip from
        # ~100% (one point is five logit units), so set the band in rate
        # space, scaled by horizon: a 90-minute horizon barely dips, a
        # 4.5-day horizon lets new joiners pull it down a few points.
        if last_rate >= 99.0:
            k = min(1.0, horizon_days / 4.5)
            qs = [100.0 - 7.0 * k, 100.0 - 3.5 * k, 100.0 - 1.5 * k,
                  100.0 - 0.5 * k, 100.0]

        # Clamp every quantile into [0.85x, 1.5x] the last rate
        # (platinum up to 3x); zero-rate trophies get a small base.
        base = max(last_rate, 0.05)
        cap = 3.0 if trophy.get("tier") == "platinum" else 1.5
        lo = 0.85 * last_rate
        hi = min(100.0, cap * base)
        out = []
        prev = 0.0
        for q in qs:
            v = round4(min(hi, max(lo, min(100.0, max(0.0, q)))))
            if v < prev:
                v = prev
            out.append(v)
            prev = v
        rows.append({"id": trophy["id"], "q": out})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
