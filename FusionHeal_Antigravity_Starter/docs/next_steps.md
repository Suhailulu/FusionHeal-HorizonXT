# Next implementation milestones

## P0 — first integration
- Run tests on every machine.
- Agree on the CSV schema and sensor naming conventions.
- Fix any dependency or environment mismatch.

## P1 — core detection
- Add timestamp-gap detection.
- Add spike, stuck-sensor, and drift detectors.
- Test legitimate operating transitions to measure false positives.

## P2 — evidence and reconstruction
- Store evidence items separately (temporal, cross-sensor, historical, process rule).
- Add candidate alternatives and diagnostics.
- Add rule IDs, severity, and unknown states to constraint results.

## P3 — evaluation
- Generate a clean synthetic baseline and inject faults into a copy.
- Preserve ground-truth fault labels and true original values.
- Split by time/episode; tune thresholds on validation only.
- Report precision, recall, F1, false-positive rate, MAE, RMSE, audit completeness, throughput.

## P4 — confidence and governance
- Keep heuristic scores labelled as heuristic.
- Only call scores probabilities after calibration and held-out validation.
- Make hard constraint failures block automatic acceptance.
- Add algorithm/config version and input hash to audit metadata.
