# FaCT Anonymous Artifact

Code and machine-verifiable evidence records for "FaCT: A Capacity-Matched, Same-Pool
Re-examination of Type-Conditioned Temporal GNNs for Attack Detection" (anonymous submission).

## Contents

- `src/` — all experiment and verification scripts (Python 3.10; see requirements.txt).
  Main pipeline: `r4_e42b_hetero.py` + `r4b_common.py` (twin-arm training/eval).
  Protocol grid + inference: `r1_train_arm.py`, `e6_inference.py`, `e9_*` (caliber aggregation).
  Calibration: `e1_window_alert.py`, `e1_hourly_readout.py`, `e5_evt_calib.py`, `e11_g3_sensitivity.py`,
  `e15_calibration_certificate.py`, `e5_direct_check.py`.
  Ablations/baselines: `e8_closure.py`, `e4_granularity.py`, `r4_ua_baseline.py`.
  Evidence chain: `verify_provenance.py` (three-gate verifier; run with no arguments for a read-only audit),
  `e17_*` ledger tooling.
- `scripts/` — smoke-test calibration script.
- `results/EXP-R2-REG/number_manifest.json` — the number manifest: maps every printed statistic
  in the paper to a frozen result-record field (path + expected value).
- `results/EXP-R2-E17/provenance_ledger.jsonl` — append-only, content-addressed evidence ledger
  (85+ rows); `provenance_first_audit.json` — latest three-gate audit output.
- `requirements.txt` — Python dependencies.

## Reproducing the three-gate audit

```
python3 src/verify_provenance.py
```
(expected output ends with `VERIFY_PROVENANCE_DONE PASS` when the artifact is intact)

## Notes

- Hard-coded absolute data paths inside scripts point to the LANL cyber-1 corpus layout used in the
  study (public, CC0, https://doi.org/10.17021/1179829); adjust `BASE`/input paths to your layout.
- Per-experiment result records (frozen JSON) ship with the released full artifact; per-seed p-values
  referenced by the paper live there.
