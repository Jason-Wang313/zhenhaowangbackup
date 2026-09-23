# FoundIR official SOW600 Qwen v8 terminal result

Completed at `2026-09-22T10:39:22Z` (`2026-09-22 18:39:22` Shanghai time).
This result supersedes every earlier official v1-v7 status and every historical
statement that Blur remains unresolved.

## Terminal result

| Category | Records | PSNR | LPIPS | SOW target | PSNR margin | LPIPS safety | Result |
|---|---:|---:|---:|---|---:|---:|---|
| Blur | 150 | 28.092635 | .144089 | >=25.9 / <=.18 | +2.192635 dB / +8.4658% | 19.9508% | PASS |
| Noise | 100 | 40.048323 | .056601 | >=38.9 / <=.10 | +1.148323 dB / +2.9520% | 43.3992% | PASS |
| Haze | 200 | 22.677420 | .110437 | >=22.1 / <=.19 | +.577420 dB / +2.6128% | 41.8750% | PASS |
| Rain | 100 | 37.809175 | .052901 | >=34.2 / <=.08 | +3.609175 dB / +10.5531% | 33.8741% | PASS |
| Low-light | 50 | 19.322459 | .148799 | >=18.9 / <=.28 | +.422459 dB / +2.2352% | 46.8577% | PASS |

`status=pass`; `all_ten_sow_checks_pass=true`.

## Protocol and provenance

- Scope: 600 authorized official records: 150 Blur, 100 Noise, 200 Haze,
  100 Rain, and 50 Low-light.
- Metrics: original-size RGB, no resize, crop, or border shave; `pyiqa` 0.1.16;
  unweighted arithmetic mean per image within each category.
- Official metric implementation:
  `third_party/foundir/cal_metrics.py`, SHA256
  `bf5bbff7c73b3a8a9623180764ff6ded6cdf4e03960d12825a5b68eaacaf9b32`.
- Router: ground-truth category was not router input; 538 correct, 62 fallback,
  zero wrong.
- Authorization:
  `reports/advisor_governance/foundir_official_sow600_qwen_authorization_2026-09-21.json`,
  SHA256 `001c70b975e6c7b3cc1683e1b9ac82e0a62d73f06020290c415ad308132fefc0`.
- Evaluation config:
  `configs/foundir_official_sow600_qwen_v8_long_singletons.json`, SHA256
  `6a84adfec7c929f291fd501976358cf61c062f35179217425b835e0d4979e04d`.
- Frozen restoration config:
  `configs/foundir_final_eleven_restoration_v9_long_singletons.json`, SHA256
  `84258615df3d806f4fac8e5dd284e44833dfaedb11946ff29a47ca26c586d191`.
- Raw local result:
  `checkpoints/foundir_official_sow600_qwen_v8_long_singletons/official_result.json`,
  SHA256 `8ea3a681eec65f1ef6b28cbd48292a17e1d6b2ee8e11f04345bd54d7cee12684`.

The previously reported development values are useful selection evidence but
are not these terminal official metrics. In particular, the terminal official
Blur result is `28.092635/.144089`, not the obsolete v1 failure
`21.8318/.38977` and not the development value `26.6147/.15697`.
