# Shibboleth

Safeguards are proxies too. An audit of the measurement validity of biosecurity
screening and bio-risk evaluation, using the Gauntlet control battery.

- `PROPOSAL.md` — thesis, the verified gap, four arms, sequencing
- `ARM_E_PROPOSAL.md` — Arm B's mechanic pointed at model-level guards rather than at a
  sequence screener. Added 2026-08-28. Arms B and E together are the paper.
- `SAFETY.md` — the non-uplifting red-teaming protocol and disclosure posture. **Read
  before any run.**
- `PREREG.md` — Arm A hypotheses, outcome measures, and gates, fixed in advance
- `FINDINGS.md` — results as they land
- `VENUES.md` — deadlines and funding

First result:

```bash
/
opt/anaconda3/bin/python3.10 scripts/run_screening_operating_point.py --scorer esm2
```

Reads gauntlet's caches in place. No downloads, no GPU, no hazardous material.
