#!/usr/bin/env python3
"""Two panels for the label-free reliability forecast.

(a) Risk-coverage. Declining to use the model on the assays with the lowest
    forecast raises mean skill on the assays it is still used on, against a
    random-abstention control at matched coverage.
(b) Cost-quality frontier for routing between ESM-2 150M and 650M on the
    forecast, against random routing and the label-using oracle.
"""
import csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_reliability_full import clusters, loco, INTERNAL, NEFF
from analyze_selective_forecast import forecast, FS

COST = 650.0 / 150.0
OUT = ROOT / "papers/ai4dd_2026/figs/forecast.pdf"


def route():
    A = {r["dms_id"]: r for r in
         csv.DictReader((ROOT / "results/reliability_full_150M.csv").open())}
    B = {r["dms_id"]: r for r in
         csv.DictReader((ROOT / "results/reliability_forecast_full.csv").open())}
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}
    ids = sorted(set(A) & set(B) & set(ref))
    g = clusters(ids, {i: ref[i]["target_seq"] for i in ids})
    gl = [g[i] for i in ids]
    ya = np.array([float(A[i]["rho_esm"]) for i in ids])
    yb = np.array([float(B[i]["rho_esm"]) for i in ids])
    F = {f: np.array([float(A[i][f]) for i in ids]) for f in INTERNAL}
    F["length"] = np.array([float(A[i]["length"]) for i in ids])
    F["n_variants"] = np.array([float(A[i]["n_variants"]) for i in ids])
    F["msa_neff_cat"] = np.array(
        [NEFF.get(ref[i].get("MSA_Neff_L_category", ""), np.nan) for i in ids])
    X = np.column_stack([F[f] for f in FS])
    ok = np.isfinite(X).all(1)
    gg = [gl[i] for i in np.where(ok)[0]]
    pred = loco(X[ok], ya[ok], gg)
    ya, yb = ya[ok], yb[ok]
    gain = yb - ya
    fr = np.linspace(0, 1, 21)

    def curve(score):
        order = np.argsort(-score)
        out = []
        for f in fr:
            k = int(round(f * len(order)))
            sel = np.zeros(len(order), bool); sel[order[:k]] = True
            skill = np.where(sel, yb, ya).mean()
            out.append(((1.0 + f * (COST - 1.0)) / COST,
                        (skill - ya.mean()) / (yb.mean() - ya.mean())))
        return np.array(out)

    rng = np.random.default_rng(0)
    rnd = np.mean([curve(rng.normal(size=len(ya)))[:, 1] for _ in range(200)], axis=0)
    r = curve(-pred)
    return r[:, 0], r[:, 1], rnd, curve(gain)[:, 1]


def main():
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))

    cov = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    style = {"150M": ("#1f77b4", "o"), "650M": ("#d62728", "s")}
    for tag, path in (("150M", "results/reliability_full_150M.csv"),
                      ("650M", "results/reliability_forecast_full.csv")):
        pred, Y, _ = forecast(path, ref)
        order = np.argsort(-pred)
        m = [Y[order[:int(round(c * len(Y)))]].mean() for c in cov]
        c, mk = style[tag]
        ax[0].plot(cov, m, marker=mk, color=c, label=f"ESM-2 {tag}, forecast")
        ax[0].axhline(Y.mean(), color=c, ls=":", lw=1)
    ax[0].set_xlabel("coverage (fraction of assays the model is used on)")
    ax[0].set_ylabel("mean Spearman skill of retained assays")
    ax[0].set_xlim(1.02, 0.48); ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title("(a) abstention on the forecast", fontsize=9)
    ax[0].text(0.62, 0.4045, "no abstention (150M)", fontsize=6.5,
               color="#1f77b4", va="bottom", ha="left")
    ax[0].text(0.62, 0.4402, "no abstention (650M)", fontsize=6.5,
               color="#d62728", va="bottom", ha="left")

    x, y, rnd, orc = route()
    ax[1].plot(x, orc, color="0.55", ls="--", lw=1, label="oracle (uses labels)")
    ax[1].plot(x, y, color="#1f77b4", lw=1.8, label="routed on the forecast")
    ax[1].plot(x, rnd, color="0.2", ls=":", lw=1.2, label="random routing")
    ax[1].axhline(1.0, color="k", lw=0.6)
    ax[1].scatter([x[4]], [y[4]], color="#d62728", zorder=5, s=22)
    ax[1].annotate("20% routed", (x[4], y[4]), textcoords="offset points",
                   xytext=(6, -10), fontsize=7)
    ax[1].set_xlabel("compute, relative to scoring everything with 650M")
    ax[1].set_ylabel("fraction of the full-650M gain captured")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("(b) routing between two model scales", fontsize=9)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        a.tick_params(labelsize=8)
        a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
