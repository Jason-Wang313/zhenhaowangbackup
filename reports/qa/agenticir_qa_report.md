# AgenticIR QA report — 2026-08-04

Status: PASS for the AgenticIR portion of the requested synthetic-data gate.

- Clean sources: 100 readable PNGs in `data/clean/`.
- Matching depth maps: 100 readable MAT files under `data/agenticir/haze/<id>/`.
- Outputs: exactly 100 each for noise, blur, haze, and rain; 400 total.
- Manifest: `manifests/agenticir_100x4.csv`, exactly 400 rows.
- Validation: every output decoded successfully, was nonempty, matched its source dimensions, and matched its recorded SHA256.
- Reproducibility: official one-image smoke outputs were repeated with fixed Python/NumPy/Torch seeds; all four hashes matched.
- Visual QC: `clean_100.jpg`, `noise_100.jpg`, `blur_100.jpg`, `haze_100.jpg`, and `rain_100.jpg` were inspected at contact-sheet scale. The effects are visible and aligned; no suspicious blank, corrupted, or dimensionally inconsistent tile was observed. No full-resolution escalation was needed.

The GenDeg portion is not passed: the official generator and structure-correction checkpoint URLs returned HTTP 404 and were quarantined as text `.part` files. Therefore no GenDeg low-light output is being represented as complete.
