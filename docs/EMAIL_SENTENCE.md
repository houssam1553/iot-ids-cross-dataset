# Turning the result into one sentence for emails

Fill the `{…}` fields from `results/key_numbers.json` **after the full run** (3 seeds). Round, don't embellish. Never use numbers from the synthetic test data.

## Rules

1. Say what you **measured**, not that you "discovered" or "proposed" something. Cross-dataset drops on these datasets are already published (see README, Related work); professors in this area will know.
2. One number for the usual result, one for yours. Two contrasts is the maximum a reader keeps.
3. Prefer detection rate / false-positive rate over F1 if F1 is hard to interpret because class proportions differ.
4. Link the repository, not a PDF, unless the preprint exists.
5. Then tie it to *their* work in the next sentence (this is what makes the email specific to them).

## Templates

**A. If re-tuning the threshold recovers most of the gap** (`share_of_gap_recovered_by_oracle_threshold` ≥ 0.5):

> I recently tested this directly: Random Forest, XGBoost and an MLP that reach F1 ≈ {temporal_within_f1} on the NetFlow v3 IoT datasets fall to F1 ≈ {cross_f1} (false alarms rising from {temporal_within_fpr}% to {cross_fpr}%) when tested on a different dataset with the same features, yet most of that gap disappears if only the decision threshold is re-tuned, which suggests the problem is largely calibration rather than lost signal (code: github.com/houssam1553/iot-ids-cross-dataset).

**B. If even the best threshold does not help much:**

> I recently tested this directly: three standard detectors that reach F1 ≈ {temporal_within_f1} on a time-ordered split of the NetFlow v3 IoT datasets drop to F1 ≈ {cross_f1} on a different dataset with identical features, and even the best possible threshold on the new network recovers only {pct}% of the gap (code: github.com/houssam1553/iot-ids-cross-dataset).

**C. If one attack family stands out** (from Table 3):

> In a small reproducible experiment on the NetFlow v3 IoT datasets, I found that detectors transfer between networks for {well_transferring_attack} (detection {x}% vs {y}%) but almost completely fail for {poorly_transferring_attack} ({a}% vs {b}%) (github.com/houssam1553/iot-ids-cross-dataset).

## Bridge sentence (adapt to each professor)

> This is why I would like to work on {domain adaptation / continual learning / explainable detection / their topic} in your group: {one concrete idea, e.g. "testing whether your X method closes this gap"}.
