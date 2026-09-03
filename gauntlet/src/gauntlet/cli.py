"""Command line interface: audit, plan, backtest."""

import argparse
import sys

import numpy as np
import pandas as pd

from . import plan as planning
from .campaign import Policy, evaluate, factorized_features
from .io import (condition_key, enumerate_single_mutants, infer_condition_columns,
                 load_campaign, select_stratum)

CHEAP = ["blosum62", "hydropathy", "n_mut"]
CHEAP_SEQ = ["mean_hydropathy", "seq_len"]
BAR = "=" * 72


def _proxy_cols(df, mode="variant"):
    cheap = CHEAP_SEQ if mode == "sequence" else CHEAP
    learned = ["esm2_seq_loglik"] if mode == "sequence" else ["esm2_wtm", "esm2_per_mut"]
    return [c for c in learned + cheap if c in df.columns]


def _prepare(args, need_candidates=False):
    df, notes = load_campaign(args.campaign, scaffold=getattr(args, "scaffold", None))
    if notes.get("unparseable"):
        print(f"! {notes['unparseable']} rows had unreadable variant labels and were dropped")
    if notes.get("mismatched"):
        print(f"! {len(notes['mismatched'])} mutations disagree with the scaffold "
              f"(e.g. {', '.join(notes['mismatched'][:4])}) — check numbering")
    if notes.get("conditions"):
        print(f"  condition columns carried through: {', '.join(notes['conditions'])}")
    cond_cols, ignored = infer_condition_columns(df, notes)
    if getattr(args, "conditions", None):
        cond_cols = [c.strip() for c in args.conditions.split(",") if c.strip() in df.columns]
        ignored = []
    if ignored:
        print("  not treated as conditions (too many distinct values): "
              + ", ".join(f"{c} ({n} levels)" for c, n in ignored))
    notes["condition_cols"] = cond_cols
    if cond_cols:
        n_strata = condition_key(df, cond_cols)[df.fitness.notna()].nunique()
        print(f"  assay conditions vary over {cond_cols} — {n_strata} stratum/strata")

    if need_candidates and not df.fitness.isna().any():
        if not notes.get("sequence"):
            raise SystemExit(
                "every row is measured and no --scaffold was given, so there are no\n"
                "candidates to choose from. Supply --scaffold to enumerate single mutants,\n"
                "or add rows with an empty fitness column.")
        cand = enumerate_single_mutants(notes["sequence"], exclude=set(df.variant))
        print(f"  enumerated {len(cand)} single mutants of the scaffold as candidates")
        df = pd.concat([df, load_campaign_frame(cand)], ignore_index=True)

    mode = notes.get("mode", "variant")
    df = planning.add_cheap_proxies(df, mode=mode)
    if getattr(args, "esm", False):
        if mode == "variant" and not notes.get("sequence"):
            raise SystemExit("--esm requires --scaffold in variant mode")
        df = planning.add_esm(df, notes.get("sequence"), notes.get("offset", 0),
                              model=args.model, batch_size=args.batch_size,
                              progress=lambda m: print(f"  {m}", flush=True), mode=mode)
    return df, notes


def load_campaign_frame(cand):
    from .io import parse_variant
    cand = cand.copy()
    cand["muts"] = cand.variant.apply(parse_variant)
    cand = cand[cand.muts.notna()]
    cand["n_mut"] = cand.muts.apply(len)
    return cand


def cmd_audit(args):
    df, notes = _prepare(args)
    df, cond_label = _enforce_conditions(df, notes, args)
    mode = notes.get("mode", "variant")
    nuis = planning.nuisance_col(mode)
    m = df[df.fitness.notna()]
    if len(m) < 5:
        raise SystemExit(f"need at least 5 measured variants to audit; found {len(m)}")
    cols = _proxy_cols(df, mode)

    where = f", condition {cond_label}" if cond_label else ""
    span = (f"{m.seq_len.min()}–{m.seq_len.max()} residues" if mode == "sequence"
            else f"{m.n_mut.min()}–{m.n_mut.max()} mutations each")
    noun = "sequences" if mode == "sequence" else "variants"
    print(f"\n{BAR}\nAUDIT — {len(m)} measured {noun}, {span}{where}\n{BAR}")
    print(f"{'proxy':16s} {'rho vs fitness':>15s} {'rho vs ' + nuis:>14s} {'partial':>9s}")
    rows = []
    for c in cols:
        r = planning._rho(m[c], m.fitness)
        cf = planning._rho(m[c], m[nuis])
        pr = _partial(m[c], m.fitness, m[nuis])
        rows.append((c, r, cf, pr))
        print(f"{c:16s} {r:>+15.3f} {cf:>+14.3f} {pr:>+9.3f}")

    learned = [r for r in rows if r[0].startswith("esm")]  # noqa: E501
    trivial = [r for r in rows if r[0] in (CHEAP_SEQ if mode == "sequence" else CHEAP)]
    if learned and trivial:
        best_l = max(learned, key=lambda r: abs(r[1]) if np.isfinite(r[1]) else -1)
        best_t = max(trivial, key=lambda r: abs(r[1]) if np.isfinite(r[1]) else -1)
        print(f"\nTrivial-baseline check: best learned proxy {best_l[0]} "
              f"({best_l[1]:+.3f}) vs best trivial {best_t[0]} ({best_t[1]:+.3f})")
        if abs(best_t[1]) >= abs(best_l[1]):
            print("  -> A scorer that knows no biology matches or beats the language "
                  "model here.\n     Do not filter on the learned score alone.")
    nm = dict((r[0], r[1]) for r in rows).get(nuis, float("nan"))
    if np.isfinite(nm) and abs(nm) >= 0.7:
        if mode == "sequence":
            print(f"\nSequence length alone predicts fitness at {nm:+.3f}. Any proxy "
                  "scored on this\n  pool is partly being credited for knowing how long "
                  "a protein is.")
        else:
            print(f"\nMutation count alone predicts fitness at {nm:+.3f}. That is campaign "
                  "progression,\n  not biology: later variants carry more edits AND better "
                  "numbers. Any proxy scored\n  on this pool is partly being credited for "
                  "knowing which round a variant came from.")

    if mode == "variant" and m.n_mut.nunique() > 1:
        scored = [r for r in rows if r[0] != nuis]
        worst = max(scored, key=lambda r: abs(r[2]) if np.isfinite(r[2]) else -1)
        if abs(worst[2]) >= planning.CONFOUND_RHO:
            print(f"\nMutation-count confound: {worst[0]} correlates {worst[2]:+.3f} with "
                  f"mutation count.\n  Summed zero-shot scores fall as edits accumulate, so on a "
                  "mixed-mutation-count pool\n  they rank the most-mutated variants last "
                  "regardless of merit.")
        _stratified(m, [c for c, *_ in scored], rows)
    elif mode == "sequence":
        print("\n  No corrected view: these are distinct proteins, so there is no "
              "mutation count\n  to hold fixed. Read the partial column, which removes "
              "sequence length.")
    print()


def _stratified(m, cols, pooled_rows, min_per_stratum=5):
    """The corrected view: correlations computed inside a fixed mutation count.

    Holding mutation count constant is the only thing measured to remove the
    confound without also removing the signal -- rescoring does not (see
    FINDINGS.md, Phase 4). Where a campaign has enough variants at a given
    count, this is the number to trust.
    """
    strata = [(k, g) for k, g in m.groupby("n_mut") if len(g) >= min_per_stratum]
    if not strata:
        print(f"\nStratified view unavailable: no mutation count has "
              f"{min_per_stratum}+ measured variants.\n  Pooled correlations above "
              "remain confounded; treat them as uninterpretable.")
        return

    covered = sum(len(g) for _, g in strata)
    print(f"\nCORRECTED VIEW — Spearman vs fitness within a fixed mutation count")
    print(f"  {len(strata)} strata covering {covered}/{len(m)} measured variants")
    head = "  " + f"{'k':>3s} {'n':>4s} " + " ".join(f"{c:>14s}" for c in cols)
    print(head)
    means = {c: [] for c in cols}
    for k, g in strata:
        cells = []
        for c in cols:
            r = planning._rho(g[c], g.fitness)
            cells.append(f"{r:>+14.3f}" if np.isfinite(r) else f"{'--':>14s}")
            if np.isfinite(r):
                means[c].append(r)
        print(f"  {int(k):>3d} {len(g):>4d} " + " ".join(cells))

    pooled = {c: r for c, r, *_ in pooled_rows}
    print("\n  " + f"{'mean':>8s} " + " ".join(
        f"{np.mean(means[c]):>+14.3f}" if means[c] else f"{'--':>14s}" for c in cols))
    print("  " + f"{'pooled':>8s} " + " ".join(
        f"{pooled[c]:>+14.3f}" if np.isfinite(pooled.get(c, np.nan)) else f"{'--':>14s}"
        for c in cols))

    if "esm2_wtm" in cols and "esm2_per_mut" in cols:
        print("\n  Note: the raw and per-mutation columns are identical inside every\n"
              "  stratum -- dividing by a constant cannot change ranks. Once you compare\n"
              "  within a fixed mutation count, per-mutation normalisation is redundant.")

    flips = [c for c in cols if means[c] and np.isfinite(pooled.get(c, np.nan))
             and np.sign(np.mean(means[c])) != np.sign(pooled[c])]
    if flips:
        print(f"\n  {', '.join(flips)} CHANGES SIGN once mutation count is held fixed.\n"
              "  The pooled number was measuring campaign progression, not the variant.")


def _warn_pooled(df, notes):
    cols = notes.get("condition_cols") or []
    if not cols:
        return
    counts = condition_key(df, cols)[df.fitness.notna()].value_counts()
    if len(counts) > 1:
        print(f"\n! Numbers below POOL {len(counts)} assay conditions "
              f"({', '.join(cols)}) and are not directly comparable.\n"
              "  `audit` and `plan` enforce a single condition; backtest does not — "
              "it replays\n  the campaign as recorded. Split the file by condition to "
              "replay one assay.")


def _partial(x, y, z):
    x, y, z = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5:
        return float("nan")
    x, y, z = x[ok], y[ok], z[ok]
    if min(np.std(x), np.std(y), np.std(z)) < 1e-12:
        return float("nan")
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y), (x, z), (y, z)))
    den = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / den) if den > 1e-9 else float("nan")


def cmd_plan(args):
    df, notes = _prepare(args, need_candidates=True)
    df, _ = _enforce_conditions(df, notes, args)
    mode = notes.get("mode", "variant")
    cols = _proxy_cols(df, mode)
    d = planning.diagnose(df, cols, mode=mode)
    policy, reasons = planning.recommend(d)

    print(f"\n{BAR}\nPLAN — budget {args.budget}\n{BAR}")
    span_label = ("sequence length" if mode == "sequence" else "mutations per variant")
    print(f"measured {d['n_measured']}   candidates {d['n_candidates']}   "
          f"{span_label} {d['k_range'][0]}–{d['k_range'][1]}")
    sup_a = d["supervised_cv_auroc"]
    sup, sup_rho = d["supervised_cv_enrich"], d["supervised_cv_rho"]
    k = planning.elite_k(d["n_measured"])
    print(f"\nHow well each scorer finds the top decile. The DECISION is made on "
          f"AUROC\n(bar {planning.TRUST_AUROC:.2f}); enrichment and Spearman are shown "
          f"because they are\neasier to act on. Scored against the top {k} of "
          f"{d['n_measured']} measured, so enrichment\nis quantised in steps of "
          f"{1.0 / (k * k / d['n_measured']):.2f}x — which is why it is not the gate.\n")
    print(f"  {'scorer':20s} {'AUROC':>8s} {'enrichment':>11s} {'spearman':>10s}")
    if np.isfinite(sup_a):
        print(f"  {'supervised (CV)':20s} {sup_a:>8.3f} {sup:>10.2f}x {sup_rho:>+10.3f}")
    else:
        print(f"  {'supervised (CV)':20s} {'n/a — too few measurements':>30s}")
    for c, a in d["proxy_auroc"].items():
        e, r = d["proxy_enrich"].get(c, np.nan), d["proxy_rho"].get(c, np.nan)
        if np.isfinite(a) or np.isfinite(e):
            acell = f"{a:>8.3f}" if np.isfinite(a) else f"{'--':>8s}"
            ecell = f"{e:>10.2f}x" if np.isfinite(e) else f"{'--':>11s}"
            rcell = f"{r:>+10.3f}" if np.isfinite(r) else f"{'--':>10s}"
            flag = "  (not selectable)" if c in planning.NOT_SELECTABLE else ""
            print(f"  {c:20s} {acell} {ecell} {rcell}{flag}")

    print(f"\nVERDICT: {policy.upper()}")
    for r in reasons:
        print(f"  - {r}")

    proxy_col = d["best_proxy"]
    if policy == "proxy" and d["mixed_k"] and "esm2_per_mut" in df.columns:
        c = d["confound"].get(proxy_col, np.nan)
        if np.isfinite(c) and abs(c) >= planning.CONFOUND_RHO:
            proxy_col = "esm2_per_mut"
            print(f"  - switching to {proxy_col} to avoid the confound")

    idx, scores = planning.select(df, policy, args.budget, proxy_col=proxy_col,
                                  beta=args.beta, seed=args.seed, mode=mode)
    keep = ["variant", "seq_len"] if mode == "sequence" else ["variant", "n_mut"]
    picks = df.loc[idx, keep].copy()
    picks["policy"] = policy
    picks["score"] = [scores.get(i, float("nan")) for i in idx]
    picks.to_csv(args.out, index=False)
    print(f"\nOrder these {len(picks)}:")
    print(picks.head(args.show).to_string(index=False))
    if len(picks) > args.show:
        print(f"  ... {len(picks) - args.show} more")
    print(f"\nwrote {args.out}")


def _enforce_conditions(df, notes, args):
    """Refuse to pool measurements taken under different assay conditions.

    Activity measured on different substrate, loading or temperature is not
    comparable -- it is why every correlation in FINDINGS.md is computed within
    a study rather than across the corpus. Pooling would credit a model for
    learning between-assay offsets and call it a fitness landscape, and it
    corrupts a correlation just as surely as it corrupts a model fit.

    Returns (subset, condition_label); the label is None when nothing varied.
    """
    cols = notes.get("condition_cols") or []
    if not cols:
        return df, None
    counts = condition_key(df, cols)[df.fitness.notna()].value_counts()
    if len(counts) <= 1:
        return df, (counts.index[0] if len(counts) else None)

    listing = "; ".join(f"{k} (n={v})" for k, v in counts.items())

    # Measure, do not assume. The old behaviour refused to pool on the strength
    # of a synthetic demonstration; on real data pooling was better and condition
    # purity was worth nothing (FINDINGS.md Phase 13).
    mode = notes.get("mode", "variant")
    pooling = None
    if not getattr(args, "condition", None):
        pooling = planning.measure_pooling(df, condition_key(df, cols), mode=mode)
    if pooling is not None:
        print(f"\nPOOLING CHECK — measured on your data, {len(pooling['table'])} "
              f"conditions with enough measurements")
        print(f"  {'within condition':22s} AUROC {pooling['within']:.3f}   "
              "(trained on that condition only)")
        print(f"  {'same size, other conds':22s} AUROC {pooling['mixed_equal_n']:.3f}   "
              "(isolates mixing from sample size)")
        print(f"  {'pooled':22s} AUROC {pooling['pooled']:.3f}   "
              "(trained on everything)")
        if pooling["purity_worthless"]:
            print("  Rows from a different condition are as useful as rows from the "
                  "same one,\n  so condition purity is buying you nothing here.")
        print(f"  -> {pooling['verdict'].upper()}: {pooling['why']}")
        if pooling["verdict"] == "pool" and not getattr(args, "pool_conditions", False):
            print("  Pooling all conditions. Use --condition to force a single one.")
            return df, None

    if getattr(args, "pool_conditions", False):
        print(f"\n! POOLING ACROSS {len(counts)} ASSAY CONDITIONS because "
              "--pool-conditions was given.\n  Strata: " + listing
              + "\n  Any number below mixes assays: a model fit is partly learning "
                "the offset\n  between them, and a correlation is partly measuring "
                "which assay a variant came from.")
        return df, None

    try:
        sub, label, _ = select_stratum(df, cols, getattr(args, "condition", None))
    except ValueError as e:
        raise SystemExit(str(e))
    kept = int(sub.fitness.notna().sum())
    dropped = int(df.fitness.notna().sum()) - kept
    print(f"\nCONDITION ENFORCEMENT — measurements span {len(counts)} assay conditions")
    print("  " + listing)
    print(f"  Using: {label}  ({kept} measured, {dropped} excluded)")
    print("  Measurements from different conditions are not comparable, so they are "
          "not pooled.\n  Override with --pool-conditions, or choose another with "
          "--condition.")
    return sub, label


MIN_BACKTEST = 20


def _replay_one(df, m, args, mode="variant"):
    """Policy replay over one set of measured variants. None if too few."""
    if len(m) < MIN_BACKTEST:
        return None
    X = planning.features(m, mode)
    y = m.fitness.to_numpy(float)

    priors = {"random": np.zeros(len(m))}
    for c in _proxy_cols(df, mode):
        priors[c] = np.nan_to_num(m[c].to_numpy(float))

    specs = [("random", Policy("random"), "random"),
             ("supervised_greedy", Policy("greedy"), "random"),
             ("supervised_ucb", Policy("ucb", beta=1.0), "random")]
    specs += [(f"rank_by_{c}", Policy("zero_shot"), c) for c in _proxy_cols(df, mode)]

    out = []
    for name, pol, pk in specs:
        r = evaluate(X, y, priors[pk], pol, args.budget, args.rounds,
                     seeds=args.seeds, top_frac=0.10)
        out.append({"policy": name, "best_found": r["best_norm"],
                    "top10pct_recall": r["top_recall"]})
    return pd.DataFrame(out).sort_values("top10pct_recall", ascending=False)


def _verdict(res, indent=""):
    base = res[res.policy == "random"].iloc[0]
    win = res[res.top10pct_recall > base.top10pct_recall].policy.tolist()
    print(f"{indent}Beat random selection: {', '.join(win) if win else 'nothing'}")
    if not win:
        print(f"{indent}  No ranking strategy helped here. Spend the next round exploring.")
    return set(win)


def _note_diagnostic_policies(winners):
    """rank_by_n_mut is a diagnostic, never a strategy.

    Inside a campaign, mutation count predicts fitness because later, more-mutated
    variants were selected to be better. Ordering by it just buys whatever carries
    the most edits, which is why `plan` excludes it from selection. Seeing it win a
    replay is evidence of campaign progression in the data, not a recommendation.
    """
    if any(w == "rank_by_n_mut" for w in winners):
        print("  NOTE: rank_by_n_mut winning is a symptom, not a strategy — inside a "
              "campaign\n  more-mutated variants were selected to be better, so it "
              "measures progression.\n  `plan` will not order by it.")


def cmd_backtest(args):
    df, notes = _prepare(args)
    cols = notes.get("condition_cols") or []
    key = condition_key(df, cols) if cols else None
    strata = key[df.fitness.notna()].value_counts() if cols is not None and cols else None

    if getattr(args, "split_conditions", False):
        if strata is None or len(strata) <= 1:
            print("\n  --split-conditions had no effect: measurements span a single "
                  "assay condition.")
        else:
            return _backtest_split(df, key, strata, args, notes.get("mode", "variant"))

    _warn_pooled(df, notes)
    m = df[df.fitness.notna()].reset_index(drop=True)
    res = _replay_one(df, m, args, notes.get("mode", "variant"))
    if res is None:
        raise SystemExit(f"need at least {MIN_BACKTEST} measured variants to "
                         f"backtest; found {len(m)}")

    print(f"\n{BAR}\nBACKTEST — {len(m)} measured variants, "
          f"budget {args.budget} x {args.rounds} rounds\n{BAR}")
    print("Replaying your campaign as if you had chosen in a different order.")
    print("Metrics: best fitness found (normalised) and recall of the true top 10%.\n")
    print(res.round(3).to_string(index=False))
    print()
    _note_diagnostic_policies(_verdict(res))
    print()


def _backtest_split(df, key, strata, args, mode="variant"):
    """Replay each assay condition separately and compare the conclusions.

    Pooling assays would let a between-assay offset masquerade as a fitness
    ranking. Replaying each separately asks the question that actually matters:
    does a policy beat random *consistently*, or did it win once by luck?
    """
    print(f"\n{BAR}\nBACKTEST BY CONDITION — {len(strata)} assay conditions, "
          f"budget {args.budget} x {args.rounds} rounds\n{BAR}")
    print("Each condition is replayed on its own; nothing is pooled.\n")

    per, skipped = {}, []
    for label, _ in strata.items():
        m = df[(key == label) & df.fitness.notna()].reset_index(drop=True)
        res = _replay_one(df, m, args, mode)
        if res is None:
            skipped.append((label, len(m)))
            continue
        per[label] = res
        print(f"--- {label}  ({len(m)} measured) ---")
        print(res.round(3).to_string(index=False))
        _verdict(res, indent="    ")
        print()

    if skipped:
        print("Skipped (fewer than "
              f"{MIN_BACKTEST} measured variants): "
              + "; ".join(f"{k} (n={n})" for k, n in skipped) + "\n")

    if len(per) < 2:
        print("Fewer than two conditions could be replayed, so no cross-condition "
              "comparison is possible.\n")
        return

    wide = pd.DataFrame({k: v.set_index("policy").top10pct_recall for k, v in per.items()})
    base = wide.loc["random"]
    beats = (wide > base).sum(axis=1)
    wide = wide.assign(**{"beats_random": [f"{b}/{len(per)}" for b in beats]})
    print("CROSS-CONDITION — recall of the true top 10%, per condition")
    print(wide.round(3).to_string())

    consistent = [p for p in wide.index if p != "random" and beats[p] == len(per)]
    never = [p for p in wide.index if p != "random" and beats[p] == 0]
    print()
    if consistent:
        print(f"Beat random in EVERY condition: {', '.join(consistent)}")
        print("  Consistency across independent assays is the claim worth acting on.")
        _note_diagnostic_policies(consistent)
    else:
        print("No policy beat random in every condition.")
        print("  A win in one assay and not another is weak evidence; treat any "
              "ranking here as unproven.")
    if never:
        print(f"Never beat random: {', '.join(never)}")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="gauntlet",
        description="Decide what protein variants to order next, and whether any "
                    "model has earned that decision.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, esm_default=False):
        sp.add_argument("--campaign", required=True, help="CSV with variant[,fitness]")
        sp.add_argument("--scaffold", help="FASTA of the wild-type sequence")
        sp.add_argument("--esm", action="store_true",
                        help="score with ESM-2 (needs torch, transformers, --scaffold)")
        sp.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
        sp.add_argument("--batch_size", type=int, default=4)
        sp.add_argument("--conditions",
                        help="comma-separated columns to treat as assay conditions "
                             "(default: inferred)")

    def condition_flags(sp):
        sp.add_argument("--condition",
                        help="which assay condition to use (substring match)")
        sp.add_argument("--pool-conditions", dest="pool_conditions",
                        action="store_true",
                        help="allow pooling measurements across assay conditions")

    a = sub.add_parser("audit", help="is your scoring function measuring anything?")
    common(a)
    condition_flags(a)
    a.set_defaults(func=cmd_audit)

    b = sub.add_parser("plan", help="choose the next batch to order")
    common(b)
    b.add_argument("--budget", type=int, required=True)
    b.add_argument("--out", default="picks.csv")
    b.add_argument("--beta", type=float, default=0.0, help="exploration weight (UCB)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--show", type=int, default=15)
    condition_flags(b)
    b.set_defaults(func=cmd_plan)

    c = sub.add_parser("backtest", help="replay your campaign to compare strategies")
    common(c)
    c.add_argument("--budget", type=int, default=8)
    c.add_argument("--rounds", type=int, default=3)
    c.add_argument("--seeds", type=int, default=300)
    c.add_argument("--split-conditions", dest="split_conditions", action="store_true",
                   help="replay each assay condition separately and compare")
    c.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
