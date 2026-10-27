"""Method check on simulated data: plant a known full-moon effect in the real panel structure and recover it.
No real outcome values are used."""
import numpy as np
import analyze as A

A.N_PLACEBO = 2000
rng = np.random.default_rng(1)
for claim, planted in [("police", 0.10), ("police", 0.0), ("mood", -0.02), ("mood", 0.0)]:
    spec, p = A.build(claim)
    w = A.real_window(p)
    dow = p.date.dt.dayofweek.to_numpy()
    if spec["kind"] == "count":
        lam = 800 * (1 + 0.1 * (dow >= 5)) * np.exp(np.log1p(planted) * w)
        p["y"] = rng.poisson(lam).astype(float)
    else:
        p["y"] = 6.0 + 0.01 * (dow >= 5) + rng.normal(0, 0.05, len(p)) + planted * w
    mu = A.baseline(spec["kind"], p)
    stat, pv = A.window_test(spec["kind"], p, mu, rng)
    est = A.full_fit(spec["kind"], p, w)
    pred = A.predictive(spec["kind"], p, rng)
    print(claim, "planted", planted, "| estimate", {k: round(float(v), 4) for k, v in est.items()},
          "| p", round(pv, 4), "| predictive beats", round(pred["beats_share_of_placebos"], 3))
