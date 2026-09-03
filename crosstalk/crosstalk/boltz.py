"""Structure-based proxy oracle via Boltz co-folding.

Why not Boltz affinity: Boltz-2's affinity head only accepts a small-molecule
binder ("must be a ligand chain (not a protein, DNA or RNA)", <=128 heavy
atoms), so it cannot score a ParD3-ParE protein-protein interface at all. A
2026 fine-tuning study (arXiv:2512.06592) adapted it for protein-protein
affinity and found it underperforms sequence-based baselines.

What is usable is co-folding confidence. ipTM and interface PAE are the proxies
binder designers actually optimize, and this landscape can audit them: every
variant has a measured on-target AND off-target fitness, so we can ask directly
whether a structural confidence score recovers specificity or only bulk
foldability. Published precedent says it will struggle -- in the Adaptyv EGFR
benchmark ipTM/interface-PAE separated real binders at roughly chance.

Sequences are from UniProt, and each was verified independently:
  ParD3  F7YBW8  Mesop_5599  93aa  -- D61/K64/E80 match the wild-type "DKE"
  ParE3  F7YBW7  Mesop_5598 103aa  -- co-operonic with ParD3 (cognate)
  ParD2  F7Y4V9  Mesop_5170  91aa  -- 39.6% identity to ParD3 (paper: 41%)
  ParE2  F7Y4W0  Mesop_5171  94aa  -- co-operonic with ParD2 (non-cognate)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# UniProt F7YBW8 / F7YBW7 / F7Y4V9 / F7Y4W0, M. opportunistum LMG 24607 (WSM2075)
PARD3 = ("MANVEKMSVAVTPQQAAVMREAVEAGEYATASEIVREAVRDWLAKRELRHDDIRRLRQLWD"
         "EGKASGRPEPVDFDALRKEARQKLTEVPPNGR")
PARE3 = ("MAVRLVWSPTAKADLIDIYVMIGSENIRAADRYYDQLEARALQLADQPRMGVRRPDIRPSA"
         "RMLVEAPFVLLYETVPDTDDGPVEWVEIVRVVDGRRDLNRLF")
PARD2 = ("MATVEKVSVALSPELLEMVKGAVDSGRYGSASEVIREALREWRLRQPLREAEAERLRKAWI"
         "EGLESGPFAPFDIEDIKQKARSRLVDAIKK")
PARE2 = ("MPIIRSPAAEGDLVDIWLAIANDSPRAADHFLDAIAERILQLAAFPESGPRRPDIGADARA"
         "LTIGNYLILYRLAEGWIEIVRIVHGARDVSTLF")

PARTNERS = {"ParE3": PARE3, "ParE2": PARE2}
MUT_POSITIONS = (61, 64, 80)   # 1-based, as reported by Lite et al.


def variant_sequence(variant: str, scaffold: str = PARD3) -> str:
    """Apply a 3-letter variant code to ParD3 at positions 61/64/80."""
    if len(variant) != len(MUT_POSITIONS):
        raise ValueError(f"variant must be {len(MUT_POSITIONS)} residues, got {variant!r}")
    s = list(scaffold)
    for aa, pos in zip(variant, MUT_POSITIONS):
        s[pos - 1] = aa
    return "".join(s)


def wild_type_code(scaffold: str = PARD3) -> str:
    return "".join(scaffold[p - 1] for p in MUT_POSITIONS)


def complex_yaml(variant: str, partner: str) -> str:
    """Boltz input for the ParD3-variant : ParE complex."""
    if partner not in PARTNERS:
        raise ValueError(f"unknown partner {partner!r}; expected one of {list(PARTNERS)}")
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {variant_sequence(variant)}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {PARTNERS[partner]}\n"
    )


@dataclass
class BoltzResult:
    variant: str
    partner: str
    iptm: float | None
    ptm: float | None
    confidence: float | None
    plddt: float | None


def parse_confidence(pred_dir: Path) -> dict:
    """Read Boltz confidence JSON for a finished prediction."""
    hits = sorted(pred_dir.glob("confidence_*_model_0.json")) or \
           sorted(pred_dir.glob("**/confidence_*_model_0.json"))
    if not hits:
        raise FileNotFoundError(f"no confidence JSON under {pred_dir}")
    return json.loads(hits[0].read_text())


class BoltzOracle:
    """Co-fold variants against a partner and return interface confidence.

    Results are cached per (variant, partner): folding is the expensive step by
    orders of magnitude, and the sweep re-reads the same variants.
    """

    def __init__(self, workdir: str | Path, boltz_bin: str = "boltz",
                 devices: int = 1, accelerator: str = "gpu",
                 use_msa_server: bool = True, fast: bool = False):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.boltz_bin = boltz_bin
        self.devices = devices
        self.accelerator = accelerator
        self.use_msa_server = use_msa_server
        # fast mode: one diffusion sample, no recycling. Measured at ~40 min per
        # 196-residue complex on CPU/MPS versus no completion in 50 min at
        # defaults, so it is the only setting that finishes without CUDA.
        self.fast = fast
        self.cache_path = self.workdir / "boltz_cache.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    @property
    def available(self) -> bool:
        return shutil.which(self.boltz_bin) is not None

    def _save(self):
        self.cache_path.write_text(json.dumps(self.cache, indent=2))

    def predict(self, variant: str, partner: str, timeout: int = 3600) -> BoltzResult:
        key = f"{variant}:{partner}"
        if key in self.cache:
            c = self.cache[key]
            return BoltzResult(variant, partner, c.get("iptm"), c.get("ptm"),
                               c.get("confidence_score"), c.get("complex_plddt"))
        if not self.available:
            raise RuntimeError(
                f"'{self.boltz_bin}' not found. Install with `pip install boltz`. "
                "Co-folding needs a GPU; CPU inference is far too slow for a sweep.")

        name = f"{variant}_{partner}"
        case = self.workdir / name
        case.mkdir(parents=True, exist_ok=True)
        yaml_path = case / f"{name}.yaml"
        yaml_path.write_text(complex_yaml(variant, partner))

        cmd = [self.boltz_bin, "predict", str(yaml_path),
               "--out_dir", str(case), "--devices", str(self.devices),
               "--accelerator", self.accelerator, "--output_format", "pdb"]
        if self.use_msa_server:
            cmd.append("--use_msa_server")
        if self.fast:
            cmd += ["--recycling_steps", "0", "--diffusion_samples", "1",
                    "--sampling_steps", "25", "--num_workers", "0"]
        subprocess.run(cmd, check=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        conf = parse_confidence(case)
        self.cache[key] = conf
        self._save()
        return BoltzResult(variant, partner, conf.get("iptm"), conf.get("ptm"),
                           conf.get("confidence_score"), conf.get("complex_plddt"))
