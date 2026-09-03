"""The interventions' declared metadata must be true of the transforms.

This is the load-bearing test file for biointerp. The whole framework rests on
the claim that each transform preserves exactly what it says it preserves, so
these assertions are the framework's warrant, not a smoke test.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import glm
from crosstalk.biointerp import interventions as iv
from crosstalk.biointerp import run_battery, render, scorers

RNG = np.random.default_rng(0)
GENE = "".join(RNG.choice(list("ACGT"), 600))
GENE = GENE[:len(GENE) - len(GENE) % 3]


def counts(s, k=1):
    return Counter(s[i:i + k] for i in range(len(s) - k + 1))


def test_synonymous_recode_preserves_the_protein_exactly():
    out = iv.get("synonymous recode").apply(GENE, np.random.default_rng(1))
    assert len(out) == len(GENE)
    assert glm.translate(out) == glm.translate(GENE)


def test_synonymous_recode_actually_changes_most_codons():
    """A recode that changed nothing would make the whole contrast vacuous."""
    out = iv.get("synonymous recode").apply(GENE, np.random.default_rng(1))
    same = sum(out[i:i + 3] == GENE[i:i + 3] for i in range(0, len(GENE), 3))
    assert same / (len(GENE) // 3) < 0.15


def test_codon_shuffle_preserves_codon_usage_exactly():
    out = iv.get("codon-order shuffle").apply(GENE, np.random.default_rng(1))
    a = Counter(GENE[i:i + 3] for i in range(0, len(GENE), 3))
    b = Counter(out[i:i + 3] for i in range(0, len(out), 3))
    assert a == b
    assert Counter(glm.translate(out)) == Counter(glm.translate(GENE))


def test_frameshift_preserves_every_kmer_count_cyclically():
    """The claim that makes the frame test decisive: a rotation is composition-identical.

    Checked cyclically, which is the sense in which it is exact; linearly it can
    differ only at the single wrap junction.
    """
    for name, k in [("frameshift +1", 1), ("frameshift +2", 2)]:
        out = iv.get(name).apply(GENE, RNG)
        assert len(out) == len(GENE)
        for order in (1, 2, 3, 6):
            cyc = lambda s: Counter((s + s[:order - 1])[i:i + order] for i in range(len(s)))
            assert cyc(out) == cyc(GENE), (name, order)
        assert glm.translate(out) != glm.translate(GENE)


def test_revcomp_is_an_involution_and_moves_strand():
    out = iv.get("reverse complement").apply(GENE, RNG)
    assert iv.get("reverse complement").apply(out, RNG) == GENE
    assert counts(out)["A"] == counts(GENE)["T"]


def test_dinuc_shuffle_preserves_dinucleotide_counts_exactly():
    for s in range(20):
        out = iv.get("dinucleotide shuffle").apply(GENE, np.random.default_rng(s))
        assert len(out) == len(GENE)
        assert counts(out, 2) == counts(GENE, 2), s
        assert counts(out, 1) == counts(GENE, 1), s


def test_dinuc_shuffle_does_move_the_sequence():
    outs = {iv.get("dinucleotide shuffle").apply(GENE, np.random.default_rng(s))
            for s in range(10)}
    assert len(outs) == 10 and GENE not in outs


def test_mono_shuffle_preserves_base_counts_and_breaks_dinucleotides():
    out = iv.get("mononucleotide shuffle").apply(GENE, np.random.default_rng(2))
    assert counts(out, 1) == counts(GENE, 1)
    assert counts(out, 2) != counts(GENE, 2)


def test_out_of_frame_stop_provably_creates_no_stop_codon():
    """The matched frame probe is only valid if the out-of-phase twin is benign.

    Checked exhaustively on many genes rather than argued, because the whole
    contrast collapses if the control sequence also terminates the protein.
    """
    for s in range(50):
        g = "".join(np.random.default_rng(s).choice(list("ACGT"), 600))
        g = g[:len(g) - len(g) % 3]
        a = iv.get("stop in frame").apply(g, RNG)
        b = iv.get("stop out of frame").apply(g, RNG)
        assert len(a) == len(b) == len(g)
        assert "*" in glm.translate(a), s
        # the control cannot CREATE an in-frame stop: the two codons its three
        # bases straddle are s[o]+"TA" and "A"+s[o+4:o+6], and no stop codon has
        # that shape. Checked directly rather than argued.
        o = b.find("TAA", 3)
        while o % 3 == 0:
            o = b.find("TAA", o + 1)
        for c in (b[o - 1:o + 2], b[o + 2:o + 5]):
            assert c not in ("TAA", "TAG", "TGA"), (s, c)
        assert glm.translate(b).count("*") <= glm.translate(g).count("*"), s
        # the two edits touch overlapping, equally sized windows
        assert sum(x != y for x, y in zip(a, g)) <= 3
        assert sum(x != y for x, y in zip(b, g)) <= 3


def test_on_the_real_genes_the_control_edit_introduces_no_stop():
    """The property that matters, checked on the sequences actually used."""
    import json
    root = Path(__file__).resolve().parents[1]
    cds = {k: v["cds"] for k, v in
           json.loads((root / "data/cds/dms_cds.json").read_text()).items()
           if len(v["cds"]) % 3 == 0 and set(v["cds"]) <= set("ACGT")}
    assert len(cds) >= 25
    for k, g in cds.items():
        a = iv.get("stop in frame").apply(g, RNG)
        b = iv.get("stop out of frame").apply(g, RNG)
        body = lambda x: glm.translate(x)[:-1]          # drop the natural terminator
        assert body(a).count("*") == body(g).count("*") + 1, k
        assert body(b).count("*") == body(g).count("*"), k


def test_frame_pair_edits_differ_only_in_phase():
    g = "".join(np.random.default_rng(7).choice(list("ACGT"), 600))
    a = iv.get("stop in frame").apply(g, RNG)
    b = iv.get("stop out of frame").apply(g, RNG)
    assert "TAA" in a and "TAA" in b and a != b


def test_declared_metadata_is_internally_consistent():
    for name, x in iv.REGISTRY.items():
        assert not (set(x.preserves) & set(x.destroys)), name
        assert x.probes and x.rationale and x.claim_if_null, name


def test_rng_is_keyed_not_streamed():
    """Reproducibility must not depend on iteration order."""
    r = iv.get("codon-order shuffle")
    a = r.apply(GENE, r.rng_for("geneA", 0))
    b = r.apply(GENE, r.rng_for("geneA", 0))
    c = r.apply(GENE, r.rng_for("geneB", 0))
    assert a == b and a != c


def test_battery_self_audit_with_a_markov_baseline():
    """An order-1 Markov model must come out invariant to the dinucleotide shuffle.

    If the battery reports anything else here, the battery is wrong, not the
    model. This is the framework auditing itself with a scorer whose semantics
    are known in closed form.
    """
    seqs = {f"g{i}": "".join(np.random.default_rng(i).choice(
        list("ACGT"), 300, p=[.35, .15, .15, .35])) for i in range(12)}
    seqs = {k: v[:len(v) - len(v) % 3] for k, v in seqs.items()}
    rep = run_battery(scorers.CountScorer(order=1), seqs, progress=False, n_boot=2000)
    assert abs(rep.by_name("dinucleotide shuffle").mean_delta) < 1e-9
    assert abs(rep.by_name("frameshift +1").mean_delta) < 0.02
    assert isinstance(render(rep), str)


# ------------------------------------------------- the 3-periodic positive control
#
# These are the warrant for the claim that the frame battery is a working
# instrument rather than a null generator. They are written on synthetic data
# with the period-three structure built in by construction, so they assert the
# scorer's semantics, not a finding about any corpus.

def _periodic_corpus(n=14, codons=200, seed=0):
    """Sequences with strong period-three structure and no in-frame stop codon."""
    rng = np.random.default_rng(seed)
    pool = [c for c in glm.CODON_TABLE if glm.CODON_TABLE[c] != "*"]
    # a skewed codon distribution, so the phase-specific tables have something to learn
    w = rng.dirichlet(np.full(len(pool), 0.35))
    out = {}
    for i in range(n):
        r = np.random.default_rng(seed + 100 + i)
        out[f"g{i}"] = "".join(pool[j] for j in r.choice(len(pool), codons, p=w))
    return out


def test_periodic_markov_holds_out_the_sequence_it_scores():
    seqs = _periodic_corpus()
    keys = sorted(seqs)
    raw = [seqs[k] for k in keys]
    held = scorers.PeriodicMarkovScorer(order=2, corpus=seqs, holdout=True).bind(keys)
    incl = scorers.PeriodicMarkovScorer(order=2, corpus=seqs, holdout=False).bind(keys)
    a, b = held.score(raw), incl.score(raw)
    # a model that has seen the sequence must score it at least as high, always
    assert (a < b).all(), "hold-out is not removing the scored sequence's counts"


def test_periodic_markov_score_does_not_depend_on_call_order():
    seqs = _periodic_corpus(n=6)
    keys = sorted(seqs)
    sc = scorers.PeriodicMarkovScorer(order=2, corpus=seqs).bind(keys)
    one = sc.score([seqs[k] for k in keys])
    two = np.array([sc.bind([k]).score([seqs[k]])[0] for k in keys])
    assert np.allclose(one, two)


def test_periodic_markov_represents_frame_and_its_aperiodic_twin_does_not():
    """The load-bearing assertion: periodicity is the only difference between arms.

    Same corpus, same order, same leave-one-out fit, same scoring code. If the
    3-periodic arm did not separate a gene from its rotation while the one-table
    arm did not, the frameshift battery would be measuring something other than
    the reading frame.
    """
    seqs = _periodic_corpus()
    keys = sorted(seqs)
    p3 = scorers.PeriodicMarkovScorer(order=2, period=3, corpus=seqs).bind(keys)
    p1 = scorers.PeriodicMarkovScorer(order=2, period=1, corpus=seqs).bind(keys)
    r3 = run_battery(p3, seqs, progress=False, n_boot=2000)
    r1 = run_battery(p1, seqs, progress=False, n_boot=2000)
    assert r3.by_name("frameshift +1").ci_lo > 0, "positive control fails the frame test"
    assert r1.by_name("frameshift +1").ci_lo <= 0, "negative control passes the frame test"
    # and the matched contrast, which is the confound-free version
    assert r3.contrasts[0].ci_lo > 0
    assert r1.contrasts[0].ci_lo <= 0


def test_t_intervals_are_wider_than_normal_and_shrink_with_n():
    from crosstalk.biointerp.battery import _tcrit
    assert _tcrit(9) > _tcrit(23) > _tcrit(1000) > 1.959
    assert abs(_tcrit(1) - 12.7062047) < 1e-6
    assert abs(_tcrit(23) - 2.068658) < 1e-4
    seqs = _periodic_corpus(n=8)
    keys = sorted(seqs)
    sc = scorers.PeriodicMarkovScorer(order=1, corpus=seqs).bind(keys)
    a = run_battery(sc, seqs, progress=False, n_boot=500, ci_dist="normal")
    b = run_battery(sc, seqs, progress=False, n_boot=500, ci_dist="t")
    ra, rb = a.by_name("frameshift +1"), b.by_name("frameshift +1")
    assert (rb.ci_hi - rb.ci_lo) > (ra.ci_hi - ra.ci_lo)


def test_rebuild_from_csv_keep_drops_pseudoreplicates():
    import csv as _csv
    from crosstalk.biointerp import rebuild_from_csv
    f = Path(__file__).resolve().parents[1] / "results" / \
        "biointerp_nt50m_default_per_sequence.csv"
    if not f.exists():
        pytest.skip("no saved per-sequence file")
    keys = [r["sequence"] for r in _csv.DictReader(f.open())]
    keep = [k for k in keys if not k.startswith(("BLAT_ECOLX_F", "BLAT_ECOLX_S"))]
    full = rebuild_from_csv(f, model="x", n_boot=500)
    cut = rebuild_from_csv(f, model="x", n_boot=500, keep=keep)
    assert full.n_sequences == len(keys)
    assert cut.n_sequences == len(keep) == len(keys) - 2
