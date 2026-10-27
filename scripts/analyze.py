"""Pre-registered full moon analysis.
Plan: https://lancedobson.com/essays/full-moon-pre-registration/

Each claim is a panel of one or more units (a city or a national series), one row per unit-day.
Counts: Poisson with unit x year-month, unit x weekday and unit x holiday fixed effects.
Mood:   linear model with the same fixed effects.
Chance: in each lunar cycle the window is moved to a random day, N_PLACEBO times.
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "raw", ROOT / "out"
N_PLACEBO = 10_000
SEED = 20261026


# ---------- holidays ----------
def nth_weekday(y, m, wd, n):
    d = date(y, m, 1)
    d += timedelta(days=(wd - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def last_weekday(y, m, wd):
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - wd) % 7)


def easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31
    return date(y, mo, (h + l - 7 * m + 114) % 31 + 1)


def holidays(y):
    tg = nth_weekday(y, 11, 3, 4)
    return {
        "new_year": date(y, 1, 1), "mlk": nth_weekday(y, 1, 0, 3), "valentine": date(y, 2, 14),
        "presidents": nth_weekday(y, 2, 0, 3), "easter": easter(y), "mothers": nth_weekday(y, 5, 6, 2),
        "memorial": last_weekday(y, 5, 0), "fathers": nth_weekday(y, 6, 6, 3), "july4": date(y, 7, 4),
        "labor": nth_weekday(y, 9, 0, 1), "halloween": date(y, 10, 31), "thanksgiving": tg,
        "black_friday": tg + timedelta(days=1), "xmas_eve": date(y, 12, 24), "xmas": date(y, 12, 25),
        "nye": date(y, 12, 31),
    }


# ---------- data ----------
def counts(name, start, end, col="all", drop=()):
    df = pd.read_csv(RAW / f"{name}.csv", parse_dates=["date"])
    s = df.set_index("date")[col]
    idx = pd.date_range(start, end, freq="D")
    s = s.reindex(idx, fill_value=0).drop(pd.to_datetime(list(drop)), errors="ignore")
    return pd.DataFrame({"date": s.index, "y": s.to_numpy(float)})


def hedonometer(series="en_all"):
    o = pd.DataFrame(json.load(open(RAW / f"hedonometer_{series}.json")))
    o["date"] = pd.to_datetime(o.date)
    return o.assign(y=o.happiness.astype(float))[["date", "y"]].sort_values("date").reset_index(drop=True)


def births():
    cdc = pd.read_csv(RAW / "US_births_1994-2003_CDC_NCHS.csv")
    ssa = pd.read_csv(RAW / "US_births_2000-2014_SSA.csv")
    b = pd.concat([cdc, ssa[ssa.year >= 2004]])
    b["date"] = pd.to_datetime(dict(year=b.year, month=b.month, day=b.date_of_month))
    return b.assign(y=b.births.astype(float))[["date", "y"]].reset_index(drop=True)


def neiss():
    return counts("neiss_daily", "2006-01-01", "2025-12-31")


NOLA_DROP = ["2014-12-31", "2015-12-31"]  # files end around noon on those days


def claims(night=False):
    c = "night" if night else "all"
    return {
        "mood": dict(kind="gauss", direction=-1, units={"hedonometer": ("utc", hedonometer)}),
        "violence": dict(kind="count", direction=1, units={
            "chicago": ("central", lambda: counts("chicago_assault", "2001-01-01", "2026-09-30", c)),
            "nyc": ("eastern", lambda: counts("nyc_assault", "2006-01-01", "2026-03-31", c))}),
        "police": dict(kind="count", direction=1, units={
            "seattle": ("pacific", lambda: counts("seattle_calls", "2010-01-01", "2026-09-30", c)),
            "nola": ("central", lambda: counts("nola_calls", "2014-01-01", "2026-09-30", c, NOLA_DROP))}),
        "er": dict(kind="count", direction=1, units={"neiss": ("central", neiss)}),
        "births": dict(kind="count", direction=1, units={"births": ("central", births)}),
    }


# ---------- moon ----------
MOON = {z: pd.read_csv(OUT / f"moon_{z}.csv", parse_dates=["date"]) for z in ("utc", "central", "eastern", "pacific")}
PHASES = pd.read_csv(OUT / "moon_phases.csv", parse_dates=["utc"])


def cycle_starts(zone):
    """Local date index (days since 1994-01-01) of every full moon, and of the new moon inside each cycle."""
    from zoneinfo import ZoneInfo
    z = {"utc": "UTC", "central": "America/Chicago", "eastern": "America/New_York", "pacific": "America/Los_Angeles"}[zone]
    origin = pd.Timestamp("1994-01-01")
    loc = PHASES.assign(d=PHASES.utc.dt.tz_convert(ZoneInfo(z)).dt.tz_localize(None).dt.normalize())
    full = loc[loc.kind == "full"].d.to_numpy()
    new = loc[loc.kind == "new"].d.to_numpy()
    f = ((full - origin.to_datetime64()) // np.timedelta64(1, "D")).astype(int)
    n = ((new - origin.to_datetime64()) // np.timedelta64(1, "D")).astype(int)
    new_off = np.array([n[(n > a) & (n < b)][0] - a for a, b in zip(f[:-1], f[1:])])
    return f[:-1], np.diff(f), new_off


# ---------- panel ----------
def build(claim, night=False):
    spec = claims(night)[claim]
    rows = []
    for unit, (zone, loader) in spec["units"].items():
        df = loader().merge(MOON[zone], on="date", how="left")
        df["unit"], df["zone"] = unit, zone
        rows.append(df)
    p = pd.concat(rows, ignore_index=True)
    p["di"] = (p.date - pd.Timestamp("1994-01-01")).dt.days
    p["ym"] = p.unit + "_" + p.date.dt.strftime("%Y-%m")
    p["dow"] = p.unit + "_" + p.date.dt.dayofweek.astype(str)
    hol = {}
    for y in range(p.date.dt.year.min(), p.date.dt.year.max() + 1):
        for k, d in holidays(y).items():
            hol[pd.Timestamp(d)] = k
    p["hol"] = p.date.map(hol)
    p["hol"] = np.where(p.hol.isna(), "none", p.unit + "_" + p.hol.astype(str))
    return spec, p


def dummies(p, cols):
    """Fixed-effect dummies with no reference collisions: year-month absorbs the unit level,
    so one weekday is dropped per unit, and holidays only flag holiday days."""
    X = pd.get_dummies(p[cols], dtype=float)
    drop = [c for c in X.columns if c.endswith("_none") or (c.startswith("dow_") and c.endswith("_0"))]
    if "ym" in cols:
        drop.append([c for c in X.columns if c.startswith("ym_")][0])
    return X.drop(columns=drop)


def design(p, extra=None):
    X = dummies(p, ["ym", "dow", "hol"])
    X.insert(0, "const", 1.0)
    if extra is not None:
        X.insert(1, "moon", extra.astype(float))
    return X


def fit(kind, y, X):
    if kind == "count":
        return sm.GLM(y, X, family=sm.families.Poisson()).fit(scale="X2")
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})


# ---------- windows ----------
def real_window(p, width=3, phase="full"):
    col = f"{phase}_day"
    w = np.zeros(len(p), bool)
    for unit, g in p.groupby("unit"):
        days = set(g.loc[g[col], "di"])
        h = width // 2
        hit = g.di.apply(lambda d: any((d + k) in days for k in range(-h, h + 1)))
        w[g.index] = hit.to_numpy()
    return w


def placebo_offsets(rng, lengths, n, margin):
    return (rng.random((n, len(lengths))) * (lengths - margin)).astype(int)


def window_sums(p, vals, width, offsets, phase_shift=None):
    """For each placebo (row of offsets) the sum of `vals` over all window days, across units.
    `vals` is a dict name -> array aligned with p. Returns dict name -> (n_placebo,) array, and window day counts."""
    out = {k: 0.0 for k in list(vals) + ["_n"]}
    h = width // 2
    for unit, g in p.groupby("unit"):
        starts, lengths, new_off = cycle_starts(g.zone.iloc[0])
        if phase_shift == "new":
            centers = starts[None, :] + (offsets + new_off[None, :]) % lengths[None, :]
        else:
            centers = starts[None, :] + offsets
        lo = g.di.min()
        span = g.di.max() - lo + 1
        pos = g.di.to_numpy() - lo
        a = np.clip(centers - lo - h, 0, span)
        b = np.clip(centers - lo + h + 1, 0, span)
        for k, v in list(vals.items()) + [("_n", None)]:
            arr = np.zeros(span)
            arr[pos] = 1.0 if v is None else v[g.index]
            cs = np.concatenate([[0.0], np.cumsum(arr)])
            out[k] = out[k] + (cs[b] - cs[a]).sum(axis=1)
    return out


# ---------- statistics ----------
def baseline(kind, p):
    X = design(p)
    m = fit(kind, p.y.to_numpy(), X)
    return m.fittedvalues.to_numpy() if kind == "count" else p.y.to_numpy() - m.resid.to_numpy()


def stat_window(kind, y, mu, sums_w, total_y, total_mu, n_all):
    """Window effect from window sums. Count: log rate ratio. Gauss: mean residual inside minus outside."""
    if kind == "count":
        return np.log(sums_w["y"] / sums_w["mu"])
    r_in = sums_w["r"] / sums_w["_n"]
    r_out = (total_y - sums_w["r"]) / (n_all - sums_w["_n"])
    return r_in - r_out


def window_test(kind, p, mu, rng, width=3, phase="full", contrast=False):
    y = p.y.to_numpy()
    vals = {"y": y, "mu": mu} if kind == "count" else {"r": y - mu}
    lengths = cycle_starts(p.zone.iloc[0])[1]
    offs = placebo_offsets(rng, lengths, N_PLACEBO, 3 if width == 3 else 1)
    zero = np.zeros((1, len(lengths)), int)

    def stat(offsets):
        if not contrast:
            s = window_sums(p, vals, width, offsets, None if phase == "full" else "new")
            return stat_window(kind, y, mu, s, (y - mu).sum() if kind != "count" else None, None, len(p))
        sf = window_sums(p, vals, width, offsets)
        sn = window_sums(p, vals, width, offsets, "new")
        if kind == "count":
            return np.log(sf["y"] / sf["mu"]) - np.log(sn["y"] / sn["mu"])
        return sf["r"] / sf["_n"] - sn["r"] / sn["_n"]

    real = stat(zero)[0]
    null = stat(offs)
    p_two = (np.sum(np.abs(null) >= abs(real)) + 1) / (len(null) + 1)
    return real, p_two


def full_fit(kind, p, w):
    m = fit(kind, p.y.to_numpy(), design(p, w))
    b, se = m.params["moon"], m.bse["moon"]
    if kind == "count":
        return dict(pct=100 * (np.exp(b) - 1), lo=100 * (np.exp(b - 1.96 * se) - 1), hi=100 * (np.exp(b + 1.96 * se) - 1))
    sd = np.std(m.resid)
    return dict(points=b, lo=b - 1.96 * se, hi=b + 1.96 * se, share_of_daily_swing=b / sd, resid_sd=sd)


def illumination_test(kind, p, mu, rng):
    """Score statistic for nightly illumination (0-1). Placebo: rotate illumination within each cycle."""
    y = p.y.to_numpy()
    x_real = p.illumination.to_numpy()
    lengths = cycle_starts(p.zone.iloc[0])[1]

    def stat(x):
        if kind == "count":
            xc = x - (x * mu).sum(-1, keepdims=True) / mu.sum()
            return (xc * (y - mu)).sum(-1) / (xc ** 2 * mu).sum(-1)
        r = y - mu
        xc = x - x.mean(-1, keepdims=True)
        return (xc * r).sum(-1) / (xc ** 2).sum(-1)

    real = stat(x_real)
    # Rotation: a day d in cycle k takes the illumination of day start_k + (d - start_k + o_k) mod L_k.
    null = []
    for _ in range(N_PLACEBO // 500):
        o = placebo_offsets(rng, lengths, 500, 1)
        X = np.empty((500, len(p)), np.float32)
        for unit, g in p.groupby("unit"):
            starts, L, _ = cycle_starts(g.zone.iloc[0])
            di = g.di.to_numpy()
            k = np.searchsorted(starts, di, side="right") - 1
            ok = (k >= 0) & (k < len(starts))
            k = np.clip(k, 0, len(starts) - 1)
            src = starts[k][None, :] + (di - starts[k] + o[:, k]) % L[k][None, :]
            lookup = MOON[g.zone.iloc[0]].set_index((MOON[g.zone.iloc[0]].date - pd.Timestamp("1994-01-01")).dt.days).illumination
            vals = lookup.reindex(src.ravel()).to_numpy().reshape(src.shape).copy()
            vals[:, ~ok] = x_real[g.index][~ok]
            X[:, g.index] = vals
        null.append(stat(X.astype(float)))
    null = np.concatenate(null)
    return real, (np.sum(np.abs(null) >= abs(real)) + 1) / (len(null) + 1)


def by_year(kind, p, mu, w, direction):
    out = []
    for yr, g in p.assign(w=w, mu=mu).groupby(p.date.dt.year):
        if g.w.sum() == 0 or (~g.w).sum() == 0:
            continue
        if kind == "count":
            e = 100 * (g.y[g.w].sum() / g.mu[g.w].sum() - 1)
        else:
            r = g.y - g.mu
            e = r[g.w].mean() - r[~g.w].mean()
        out.append(dict(year=int(yr), effect=float(e), claimed_direction=bool(np.sign(e) == direction)))
    k = sum(o["claimed_direction"] for o in out)
    from scipy.stats import binomtest
    return dict(years=out, n_years=len(out), n_claimed=int(k), sign_test_p=float(binomtest(k, len(out), 0.5).pvalue))


# ---------- predictive ----------
def predictive(kind, p, rng):
    p = p.sort_values(["unit", "date"]).reset_index(drop=True)
    y = p.y.to_numpy()
    trail = p.groupby("unit").y.transform(lambda s: s.shift(1).rolling(28, min_periods=20).mean()).to_numpy()
    cut = p.date.sort_values().iloc[int(len(p) * 2 / 3)]
    cut = pd.Timestamp(f"{cut.year + (cut.month > 6)}-01-01")
    ok = ~np.isnan(trail)
    train, test = ok & (p.date < cut).to_numpy(), ok & (p.date >= cut).to_numpy()
    Xd = dummies(p, ["dow", "hol"])
    if kind == "count":
        m = sm.GLM(y[train], sm.add_constant(Xd[train]), family=sm.families.Poisson(), offset=np.log(trail[train])).fit()
        base = np.exp(np.log(np.where(ok, trail, 1)) + sm.add_constant(Xd).to_numpy() @ m.params.to_numpy())
    else:
        m = sm.OLS(y[train] - trail[train], sm.add_constant(Xd[train])).fit()
        base = np.where(ok, trail, 0) + sm.add_constant(Xd).to_numpy() @ m.params.to_numpy()
    lengths = cycle_starts(p.zone.iloc[0])[1]
    offs = np.vstack([np.zeros((1, len(lengths)), int), placebo_offsets(rng, lengths, N_PLACEBO, 3)])
    tr, te = train.astype(float), test.astype(float)
    if kind == "count":
        vals = {"y_tr": y * tr, "b_tr": base * tr, "y_te": y * te, "b_te": base * te, "n_te": te}
    else:
        r = y - base
        vals = {"r_tr": r * tr, "n_tr": tr, "r_te": r * te, "n_te": te}
    s = window_sums(p, vals, 3, offs)
    if kind == "count":
        beta = np.log(s["y_tr"] / s["b_tr"])
        n_te = te.sum()
        Yt, Bt = y[test].sum(), base[test].sum()
        c = (np.exp(beta) * s["n_te"] + (n_te - s["n_te"])) / n_te
        ll_y = beta * s["y_te"] - np.log(c) * Yt
        ll_b = (np.exp(beta) * s["b_te"] + (Bt - s["b_te"])) / c - Bt
        gain = ll_y - ll_b
    else:
        r = y - base
        R_tr, N_tr = r[train].sum(), train.sum()
        beta = s["r_tr"] / s["n_tr"] - (R_tr - s["r_tr"]) / (N_tr - s["n_tr"])
        n = test.sum()
        wbar = s["n_te"] / n
        gain = 2 * beta * (s["r_te"] - wbar * r[test].sum()) - beta ** 2 * n * wbar * (1 - wbar)
    real, null = gain[0], gain[1:]
    return dict(train_until=str((cut - pd.Timedelta(days=1)).date()), test_from=str(cut.date()),
                real_gain=float(real), beats_share_of_placebos=float(np.mean(real > null)),
                predictive=bool(np.mean(real > null) >= 0.95))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def run(primary=("mood", "violence", "police", "er")):
    rng = np.random.default_rng(SEED)
    res = {"primary": {}, "secondary": {}}
    cache = {}
    for c in list(primary) + ["births"]:
        spec, p = build(c)
        mu = baseline(spec["kind"], p)
        cache[c] = (spec, p, mu)
    for c in primary:
        spec, p, mu = cache[c]
        w = real_window(p)
        real, pv = window_test(spec["kind"], p, mu, rng)
        res["primary"][c] = dict(n_days=len(p), units=list(spec["units"]), window_share=float(w.mean()),
                                 stat=float(real), p_placebo=float(pv), **full_fit(spec["kind"], p, w),
                                 by_year=by_year(spec["kind"], p, mu, w, spec["direction"]),
                                 predictive=predictive(spec["kind"], p, rng))
        print(c, {k: v for k, v in res["primary"][c].items() if k not in ("by_year", "predictive")}, file=sys.stderr)
    adj = holm(np.array([res["primary"][c]["p_placebo"] for c in primary]))
    for c, a in zip(primary, adj):
        res["primary"][c]["p_holm"] = float(a)
    sec = {}
    for c in primary:
        spec, p, mu = cache[c]
        k = spec["kind"]
        r, pv = window_test(k, p, mu, rng, width=1)
        sec[f"{c}:exact_day"] = dict(stat=float(r), p_placebo=float(pv), **full_fit(k, p, real_window(p, 1)))
        r, pv = illumination_test(k, p, mu, rng)
        sec[f"{c}:illumination"] = dict(stat=float(r), p_placebo=float(pv))
        r, pv = window_test(k, p, mu, rng, contrast=True)
        sec[f"{c}:full_vs_new"] = dict(stat=float(r), p_placebo=float(pv))
    for c in ("violence", "police"):
        if c not in primary:
            continue
        spec, p = build(c, night=True)
        mu = baseline(spec["kind"], p)
        w = real_window(p)
        r, pv = window_test(spec["kind"], p, mu, rng)
        sec[f"{c}:night"] = dict(stat=float(r), p_placebo=float(pv), **full_fit(spec["kind"], p, w))
    spec, p, mu = cache["births"]
    w = real_window(p)
    r, pv = window_test(spec["kind"], p, mu, rng)
    sec["births:window"] = dict(stat=float(r), p_placebo=float(pv), **full_fit(spec["kind"], p, w))
    adj = holm(np.array([v["p_placebo"] for v in sec.values()]))
    for (k, v), a in zip(sec.items(), adj):
        v["p_holm"] = float(a)
    res["secondary"] = sec
    return res


if __name__ == "__main__":
    prim = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("mood", "violence", "police", "er")
    out = run(prim)
    json.dump(out, open(OUT / "results.json", "w"), indent=1, default=float)
    print("written out/results.json", file=sys.stderr)
