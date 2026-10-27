"""EXPLORATORY, not pre-registered: split the police full-vs-new result by city and look at the new-moon window alone.
Writes out/exploratory_police.csv."""
import csv
import numpy as np
import analyze as A

A.N_PLACEBO = 5000
rng = np.random.default_rng(7)
spec, p = A.build("police")
rows = []
for unit in ["seattle", "nola", "both"]:
    q = p if unit == "both" else p[p.unit == unit].reset_index(drop=True)
    mu = A.baseline("count", q)
    for name, kw in [("full_window", {}), ("new_window", {"phase": "new"}), ("full_vs_new", {"contrast": True})]:
        stat, pv = A.window_test("count", q, mu, rng, **kw)
        rows.append(dict(unit=unit, test=name, pct=round(100 * np.expm1(stat), 2), p_placebo=round(pv, 4)))
        print(rows[-1])
with open(A.OUT / "exploratory_police.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
