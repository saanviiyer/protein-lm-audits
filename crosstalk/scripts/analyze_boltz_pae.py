"""Interface PAE as a specificity proxy, versus the scalar ipTM.

ipTM saturated: 0.942-0.968 against ParE3 and 0.908-0.945 against ParE2, so it
compressed the whole landscape into three points. Interface PAE is per-residue-
pair and is not bounded the same way, so it has room to separate complexes that
ipTM calls equally good.

Two variants are reported. Mean interface PAE averages every cross-chain pair,
including the many that are nowhere near the binding site. Min-k interface PAE
averages only the most confident cross-chain pairs, which is closer to "how well
is the actual interface resolved".

Lower PAE is better, so scores are negated to keep "larger is better".
"""
import csv, glob, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosstalk.boltz import PARD3, PARTNERS
from crosstalk.landscape import load_pard3
from run_proxy_ladder import auc, spearman

LA = len(PARD3)


def interface_pae(path: str, k_frac: float = 0.1) -> tuple[float, float]:
    pae = np.load(path)["pae"]
    cross = np.concatenate([pae[:LA, LA:].ravel(), pae[LA:, :LA].ravel()])
    kth = max(1, int(k_frac * cross.size))
    return float(cross.mean()), float(np.sort(cross)[:kth].mean())


def main():
    L = load_pard3()
    rec = {}
    for f in sorted(glob.glob("results/boltz/*/boltz_results_*/predictions/*/pae_*.npz")):
        name = Path(f).parent.name              # e.g. DWE_ParE3
        variant, partner = name.rsplit("_", 1)
        m, mk = interface_pae(f)
        rec.setdefault(variant, {})[partner] = (m, mk)

    variants = [v for v, d in rec.items() if set(d) == set(PARTNERS)]
    print(f"{len(variants)} variants with both partners folded\n")
    w3 = np.array([L.F[L.index[v], 0] for v in variants])
    w2 = np.array([L.F[L.index[v], 1] for v in variants])

    mean3 = np.array([rec[v]["ParE3"][0] for v in variants])
    mean2 = np.array([rec[v]["ParE2"][0] for v in variants])
    mink3 = np.array([rec[v]["ParE3"][1] for v in variants])
    mink2 = np.array([rec[v]["ParE2"][1] for v in variants])

    print("DYNAMIC RANGE (the reason for trying this)")
    print(f"  ipTM        ParE3 0.942-0.968 (span 0.026)   [from the confidence JSONs]")
    print(f"  mean iPAE   ParE3 {mean3.min():.2f}-{mean3.max():.2f} (span {np.ptp(mean3):.2f})"
          f"   ParE2 {mean2.min():.2f}-{mean2.max():.2f} (span {np.ptp(mean2):.2f})")
    print(f"  min-k iPAE  ParE3 {mink3.min():.2f}-{mink3.max():.2f} (span {np.ptp(mink3):.2f})"
          f"   ParE2 {mink2.min():.2f}-{mink2.max():.2f} (span {np.ptp(mink2):.2f})\n")

    print("CORRELATION WITH MEASURED FITNESS (lower PAE is better, so scores negated)")
    print(f"{'proxy':22s} {'vs on-target':>13s} {'vs off-target':>14s} {'vs margin':>11s}")
    for nm, s3, s2 in (("mean iPAE", mean3, mean2), ("min-k iPAE", mink3, mink2)):
        print(f"{nm:22s} {spearman(-s3, w3):+13.3f} {spearman(-s2, w2):+14.3f} "
              f"{spearman(s2 - s3, w3 - w2):+11.3f}")
    print(f"{'ipTM (for reference)':22s} {'+0.471':>13s} {'+0.335':>14s} {'+0.268':>11s}")

    spec = (w3 >= 0.8) & (w2 <= 0.2)
    prom = (w3 >= 0.8) & (w2 >= 0.6)
    mask = spec | prom
    print(f"\nDiscrimination set: {int(spec.sum())} specific vs {int(prom.sum())} promiscuous"
          f" -- n={int(mask.sum())}, far too small for AUC; shown only for completeness")
    if mask.sum() >= 4:
        for nm, s3, s2 in (("mean iPAE margin", mean3, mean2), ("min-k iPAE margin", mink3, mink2)):
            print(f"  AUC {nm:20s} {auc((s2 - s3)[mask], spec[mask]):.3f}")

    with open("results/boltz_pae.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "W_ParE3", "W_ParE2", "mean_ipae_ParE3", "mean_ipae_ParE2",
                    "mink_ipae_ParE3", "mink_ipae_ParE2"])
        for i, v in enumerate(variants):
            w.writerow([v, w3[i], w2[i], mean3[i], mean2[i], mink3[i], mink2[i]])
    print("\nwrote results/boltz_pae.csv")


if __name__ == "__main__":
    main()
