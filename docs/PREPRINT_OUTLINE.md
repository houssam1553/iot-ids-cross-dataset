# Short preprint (4-6 pages) - outline

**Working title:** *How far do IoT intrusion detectors travel? A leakage-audited cross-dataset evaluation on NetFlow v3*

Discuss authorship with Dr Boulaiche before posting (supervisor as co-author is usual).

## Where to post

* **arXiv (cs.CR or cs.LG):** first-time submitters usually need an endorsement from an existing arXiv author in that category. Ask your supervisor or a co-author.
* **Zenodo:** connect the GitHub repo to get a DOI for a tagged release (no endorsement needed). Good to do anyway, so the code you cite in emails is a frozen version.
* **TechRxiv** (IEEE) is another option without endorsement.

## Sections

1. **Introduction (½ page).** Near-perfect scores in IoT IDS papers; random within-dataset splits; question: what happens on later traffic and on a different network with the same features? Contributions as a bullet list: (i) leakage audit; (ii) random vs temporal vs cross-dataset comparison for RF/XGB/MLP; (iii) threshold-vs-ranking decomposition; (iv) per-attack transfer; (v) open code.
2. **Related work (½ page).** NetFlow datasets (Sarhan et al.), temporal v3 release (Luay et al.), cross-domain NIDS evaluation (Layeghy & Portmann), pitfalls in ML security (Arp et al.). State clearly what is reproduced and what is added.
3. **Data and leakage audit (1 page).** Table of datasets; excluded columns and why; results of `audit.md` (single-feature AUCs, duplicates, train/test overlap).
4. **Protocol (½ page).** Splits, sampling caps, models, threshold on source validation, metrics, seeds.
5. **Results (1-1½ pages).** Table 1 (headline), Figure (heatmaps), Table 3 (per-attack). Ablations: `--drop-ttl`, `--dedup`.
6. **Discussion and limitations (½ page).** Single lab origin; caps alter priors; binary task; no adaptation.
7. **Conclusion (3-4 lines)** + link to code.

## Abstract template (fill after running)

> Machine-learning intrusion detectors for IoT networks often report near-perfect accuracy, but are usually trained and tested on random splits of a single dataset. Using the NetFlow v3 versions of ToN-IoT, BoT-IoT and UNSW-NB15, which share an identical 53-feature format, we evaluate Random Forest, XGBoost and an MLP under random, time-ordered and cross-dataset splits, after excluding identifiers and timestamps and auditing for duplicate-flow leakage. F1 is {a} under random splits, {b} under time-ordered splits and {c} across datasets, while the false-positive rate rises from {d}% to {e}%. {One sentence on the threshold decomposition or per-attack result.} Code and full results are openly available.
