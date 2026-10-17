# Do IoT intrusion detectors survive a change of network?

**A small, reproducible cross-dataset test of ML-based intrusion detection on the NetFlow v3 IoT datasets.**

Houssam Zitoune · PhD student, Computer Science, Université 20 Août 1955 Skikda (Algeria) · supervised by Dr Mehdi Boulaiche

---

## Key finding

<!-- FINDING:START -->
> *Not yet filled in. Run the pipeline (see [Reproduce](#reproduce)); `python -m src.report` writes the finding here from the actual numbers.*
<!-- FINDING:END -->

## Question

ML intrusion detectors for IoT networks routinely report 99%+ accuracy. Almost always, they are trained and tested on random splits of **one** dataset. A deployed detector, however, meets a network it was not trained on.

This repository measures how much three standard models lose when they are:

1. tested on a **time-ordered** split of the same dataset (later traffic than they were trained on), and
2. tested on a **different dataset** that uses exactly the same features.

## Data

The University of Queensland's NetFlow datasets re-extract the same flow features from the original packet captures of several well-known intrusion datasets, so a model trained on one can be applied to another without any feature mapping. We use **version 3** (53 NetFlow features), because v3 adds flow start/end timestamps that make a time-ordered split possible.

| Short name | File | Flows (official) | Attack share | Classes | Network |
|---|---|---|---|---|---|
| `ton_iot` | NF-ToN-IoT-v3 | 27,520,260 | 39.0% | Benign + 9 attacks | IoT/IIoT testbed |
| `bot_iot` | NF-BoT-IoT-v3 | ~16.9M | 99.7% | Benign + 4 attacks | IoT botnet testbed |
| `unsw_nb15` | NF-UNSW-NB15-v3 | 2,365,424 | 5.4% | Benign + 9 attacks | conventional enterprise (non-IoT control) |

Counts from the [official UQ page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/). For NF-BoT-IoT-v3 the page's stated total (16,993,808) differs from the sum of its own class table (16,933,844); `src/prepare.py` prints the real counts of whatever file you downloaded and warns about any difference from the official class counts.

**Download.** Official links are on the UQ page above (each dataset also has a DOI record on UQ eSpace / researchdata.edu.au). Mirrors exist on Kaggle (e.g. `athena21/netflow-v3-datasets`); if you use one, check the count warnings from `prepare.py`. Put the CSVs anywhere under `data/raw/`; file names are matched loosely (`*ton*iot*v3*.csv`, etc.).

**Do they really share features?** Yes, by construction; this is the point of the NetFlow conversion. The pipeline still checks it: `results/feature_check.csv` lists every column, its dtype in each dataset, and whether it is used or why not.

**Caveat on independence.** All three source datasets were produced at the UNSW Canberra Cyber Range. "Different dataset" therefore means a different testbed, traffic generator and attack campaign, but not a different organisation. A drop between them is, if anything, an optimistic estimate of what happens on a truly foreign network.

## What the models are not allowed to see

| Excluded | Why |
|---|---|
| `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT` | Identify machines and services of one testbed (e.g. "the attacker is 192.168.100.x"). |
| `DNS_QUERY_ID` | Random transaction id, no behavioural meaning. |
| `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS` | Attacks were run in scheduled time windows, so time alone predicts the label. Used **only** to order the split. |
| `Label`, `Attack` | Targets. |
| `MIN_TTL`, `MAX_TTL` (optional, `--drop-ttl`) | Ablation: TTL often reflects the testbed's operating systems. |

`src/audit.py` checks the effect of these choices directly: it trains a depth-3 tree on **one column at a time** (identifiers and timestamps included on purpose) and reports the ones that alone nearly separate attacks from benign traffic. It also measures exact duplicate flows and how many test flows also appear, identically, in the training set; see `results/audit.md`.

## Protocol

```
for split in {random, temporal}:
  for seed in {0, 1, 2}:
    split every dataset 70 / 15 / 15
    for model in {Random Forest, XGBoost, MLP}:
      for source dataset S:
        train on S.train
        choose threshold on S.validation (max F1), then freeze it
        evaluate on the test split of EVERY dataset (S itself = "within", others = "cross")
```

* **Random split:** stratified by class. The usual protocol, included as the optimistic reference.
* **Temporal split:** within each class, flows are sorted by start time; earliest 70% train, next 15% validation, last 15% test. Splitting per class keeps every attack type in the test set (a single global time cut would not, because attacks ran in blocks).
* The test split of a dataset is **identical** whether the model was trained on it or on another dataset, so within-vs-cross differences come only from the training data.
* **Sampling:** to fit on a laptop, `prepare.py` keeps at most 150,000 benign flows and 30,000 flows per attack class from each dataset (random, order-preserving, seed 42). This changes class proportions, so the metrics below that do not depend on proportions (detection rate, false-positive rate, ROC-AUC) are the most comparable across datasets.
* **Models** use standard, un-tuned settings (`src/models.py`). The question is how much *any* reasonable model loses, not which model wins.

**Metrics** (attack = positive class): F1, precision, detection rate (recall on attacks), false-positive rate (benign flows flagged), balanced accuracy, MCC, ROC-AUC, PR-AUC, and the detection rate of **each attack class**. We also report an **oracle F1**, computed with the best threshold on the test set itself. It is not achievable in deployment; comparing it with the real F1 shows whether a model still *ranks* flows correctly on the new network and only its threshold is wrong, or whether the signal itself is lost.

## Results

<!-- RESULTS:START -->
*Not yet run on the real datasets. After running the pipeline, `python -m src.report` replaces this block with Tables 1-3 and the heatmap figure.*
<!-- RESULTS:END -->

Full outputs: `results/results_all.csv` (one row per model x seed x source x target x split), `results/results_per_class.csv`, `results/audit.md`, `results/key_numbers.json`.

## Reproduce

Python 3.10+. Works on a laptop (16 GB RAM recommended) or in Google Colab (`notebooks/colab_run.ipynb`).

```bash
git clone https://github.com/houssam1553/iot-ids-cross-dataset
cd iot-ids-cross-dataset
pip install -r requirements.txt

# put NF-ToN-IoT-v3.csv, NF-BoT-IoT-v3.csv, NF-UNSW-NB15-v3.csv under data/raw/
python -m src.prepare          # sample the big CSVs (slowest step: reads ~45M rows twice)
python -m src.audit            # leakage audit -> results/audit.md
python -m src.run_experiments --quick   # 1 seed, small - check everything works
python -m src.run_experiments  # full run: 3 seeds x 3 models x 2 splits
python -m src.report           # tables, figure, key finding -> README
```

Useful options: `--drop-ttl` (ablation), `--dedup` (remove exact duplicate flows before splitting), `--datasets ton_iot bot_iot` (subset), `--models rf xgb`.

`pytest -q` runs an end-to-end test on small **synthetic** files (`tests/make_synthetic.py`). Those numbers are meaningless and are never written to this README.

## Limitations

* Three datasets, all generated in one lab's cyber range; findings may not hold for production IoT traffic.
* Class caps change class proportions, so F1 and precision on a target depend on the sampled attack share; detection rate and FPR do not.
* Binary detection only. Attack families differ between datasets (e.g. BoT-IoT has no XSS), so for each target class we report detection rate, not multi-class accuracy.
* No hyper-parameter tuning, no domain adaptation. Those are the obvious next steps, not part of this measurement.

## Related work

Cross-dataset evaluation on these NetFlow datasets is not new: the UQ group that released them has evaluated it (e.g. Sarhan, Layeghy & Portmann; Layeghy & Portmann 2023, *Explainable cross-domain evaluation of ML-based network intrusion detection systems*). This repository is a **transparent, leakage-audited reproduction** on the newer v3 release, adding a time-ordered split, a threshold-vs-ranking decomposition and per-attack transfer rates.

* M. Sarhan, S. Layeghy, N. Moustafa, M. Portmann. *NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems.* BDTA 2020 / Springer 2021.
* M. Sarhan, S. Layeghy, M. Portmann. *Towards a Standard Feature Set for Network Intrusion Detection System Datasets.* Mobile Networks and Applications, 2022.
* M. Luay, S. Layeghy, S. Hosseininoorbin, M. Sarhan, N. Moustafa, M. Portmann. *Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems.* arXiv:2503.04404 (IEEE Access, 2026). **Cite this for the v3 datasets.**
* S. Layeghy, M. Portmann. *Explainable cross-domain evaluation of ML-based network intrusion detection systems.* 2023.
* D. Arp et al. *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security 2022.

## Citation

```bibtex
@misc{zitoune2026crossdataset,
  author = {Houssam Zitoune},
  title  = {Do IoT intrusion detectors survive a change of network? A cross-dataset test on NetFlow v3},
  year   = {2026},
  url    = {https://github.com/houssam1553/iot-ids-cross-dataset}
}
```

MIT licence (code). The datasets keep their own licences and citation requirements.
