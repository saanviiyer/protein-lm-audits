# Safety protocol

This project audits biosecurity safeguards. That makes the protocol part of the
method, not paperwork attached to it. The protocol is also the reason the work is
publishable in the open at all, so it belongs in the paper.

## The governing rule

**Non-uplifting red-teaming: a safeguard may be probed only with perturbations that
monotonically decrease the hazard of the input, or with inputs that carry no hazard
to begin with.**

Every experiment must fall into one of two admissible classes:

- **Class B (benign).** The input is a sequence with no hazard potential — housekeeping
  proteins, model-organism sequences, characterized enzymes from the DMS corpora. The
  measurement is the screener's behaviour on material it should never flag.
- **Class D (destructive).** The input is derived from another sequence by an operation
  that provably removes function while holding a nuisance variable fixed —
  composition-preserving shuffling, reversal, residue-frequency-matched resampling. The
  derived sequence encodes nothing. It is strictly less hazardous than its source.

Anything that is neither B nor D is out of scope for this project regardless of how
informative it would be.

## What is explicitly excluded

- **No optimization toward evasion.** No search, no gradient, no evolutionary loop, no
  language-model sampling whose objective is a lower screening score. This is the single
  hardest line and it is absolute. It rules out the most obvious experiment in the space
  and that is intentional — see below.
- **No generation of variants of sequences of concern**, by any method, including as an
  intermediate that is discarded.
- **No hazardous sequence data on this machine.** The NIST benchmark set contains
  positives derived from agents of concern. It is publicly downloadable and this project
  does not download it. Class B panels are built from benign sources only.
- **No reconstruction of hazard databases.** The screener's biorisk HMMs define what it
  considers concerning. Analysis treats them as an opaque decision function; the project
  does not characterize, enumerate, or publish their contents.
- **No release of any artifact that lowers the cost of misuse**, including intermediate
  files, per-sequence scores on hazardous input, or a ranked list of screener blind spots.

## Why the exclusion is a feature

The obvious experiment — optimize until the screener stops flagging — has already been
done once, properly, by people with the institutional apparatus to do it safely:
Wittmann et al. (*Science* 390(6768):82–87, 2025) ran it under coordinated disclosure
with IBBIS, a tiered controlled-access system, an independent review process for access
requests, and a funded endowment for perpetual hosting. Reproducing that on a laptop
would be worse science and worse practice.

The constraint forces a different and more useful question. Evasion work measures
*whether* a screener can be beaten. This project measures *what the screener is reading*
— which is the thing that determines whether it can be fixed, and the thing nobody has
published. Specificity on benign input is the half of the problem that needs no
hazardous material and that no one is working on.

## Disclosure posture

Findings here are about a **deployed tool**, so this is coordinated vulnerability
disclosure, not ordinary publication.

1. **Notify before posting.** Any result showing a specific exploitable weakness in
   `commec` goes to IBBIS (Tessa Alexanian, Nicole Wheeler) and to the NIST program
   contact (Sheng Lin-Gibson, Biosystems and Biomaterials Division) before any preprint.
2. **Default embargo 90 days** from notification, extendable at the maintainers' request.
   Say so in the paper.
3. **Aggregate over per-sequence.** Report distributions, effect sizes, and mechanisms.
   Do not publish the per-sequence table of which benign inputs slipped through or which
   nuisance feature best predicts a miss.
4. **Precedent to follow:** Sherman et al. (arXiv:2512.09233, NDSS 2026) audited
   SecureDNA's security design as external researchers, found architectural flaws, and
   the maintainers shipped mitigations in v1.1.0. That is the model — external audit,
   coordinated, and the tool got better.
5. **The specificity framing is disclosure-friendly.** A false
-positive result tells an attacker nothing useful. It tells a synthesis provider where
   its costs are coming from. That asymmetry is why this arm can be published on a
   normal timeline.

## Institutional review

This is computational work on public, benign sequence data with no human subjects, no
pathogens, and no wet lab, so it is not IRB or IBC territory. It should still be
**disclosed to a faculty advisor before any run**, and the advisor named in the paper's
responsible-use statement. Several target venues require such a statement; write it
from this file.

## Review trigger

Stop and re-read this document before any experiment that:

- introduces a sequence source not provably Class B,
- introduces any search or optimization loop over sequences,
- would produce a per-sequence artifact intended for release, or
- extends the audit to a screener other than `commec`.
