import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk.boltz import (PARD3, PARE2, PARE3, MUT_POSITIONS, BoltzOracle,
                             complex_yaml, variant_sequence, wild_type_code)
from crosstalk.landscape import load_pard3


def test_scaffold_wildtype_matches_the_landscape():
    """The UniProt ParD3 must carry DKE at 61/64/80, or it is the wrong protein.

    This is the check that identified F7YBW8 among many ParD paralogs.
    """
    assert wild_type_code() == "DKE"
    assert load_pard3().wt == wild_type_code()


def test_sequence_lengths_match_uniprot():
    assert (len(PARD3), len(PARE3), len(PARE2)) == (93, 103, 94)


def test_variant_substitutes_exactly_the_mutated_positions():
    v = variant_sequence("WYT")
    assert sum(a != b for a, b in zip(v, PARD3)) == 3
    for aa, pos in zip("WYT", MUT_POSITIONS):
        assert v[pos - 1] == aa
    assert variant_sequence("DKE") == PARD3   # wild type is a no-op


def test_variant_length_is_preserved():
    assert len(variant_sequence("AAA")) == len(PARD3)


def test_complex_yaml_contains_both_chains():
    y = complex_yaml("DYE", "ParE2")
    assert variant_sequence("DYE") in y
    assert PARE2 in y and PARE3 not in y
    assert "id: A" in y and "id: B" in y


def test_unknown_partner_rejected():
    with pytest.raises(ValueError):
        complex_yaml("DKE", "ParE9")


def test_bad_variant_length_rejected():
    with pytest.raises(ValueError):
        variant_sequence("DK")


def test_oracle_reports_availability_without_crashing(tmp_path):
    o = BoltzOracle(tmp_path)
    assert isinstance(o.available, bool)
    if not o.available:
        with pytest.raises(RuntimeError, match="not found"):
            o.predict("DKE", "ParE3")


def test_oracle_uses_cache_instead_of_folding(tmp_path):
    o = BoltzOracle(tmp_path)
    o.cache["DKE:ParE3"] = {"iptm": 0.87, "ptm": 0.8,
                            "confidence_score": 0.85, "complex_plddt": 0.9}
    r = o.predict("DKE", "ParE3")      # must not attempt to run boltz
    assert r.iptm == 0.87 and r.partner == "ParE3"
