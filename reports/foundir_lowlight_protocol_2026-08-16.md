# Official FoundIR-Lowlight acquisition and protocol

Date: 2026-08-16 UTC

## Outcome

The official FoundIR test release was acquired and validated. The Phase 1
`FoundIR-Lowlight` benchmark is frozen as the isolated Lowlight category
`14Lowlight`, containing exactly 50 paired images with IDs `1101` through
`1150`. This milestone establishes the data and protocol only. No restoration
model was run on these images, no parameter was tuned against them, and no
FoundIR performance or Phase 1 pass is claimed here.

## Authoritative sources

- repository: `House-Leo/FoundIR`
- repository revision: `6d19958499622d90ca71fe971d83b1e428c17cd0`
- repository license: MIT
- `Testset_meta_info.txt` SHA-256:
  `15467efd89a55d70a381f813d29f84b4c63ae5a958ef67ffa9bcf1883deb3761`
- `cal_metrics.py` SHA-256:
  `bf5bbff7c73b3a8a9623180764ff6ded6cdf4e03960d12825a5b68eaacaf9b32`
- paper PDF SHA-256:
  `96c94e39b6bdd1347117fe999633aeb9e38ad5d2bb92728a373580e3fe5e5ed2`
- supplement PDF SHA-256:
  `53a20bdcfa6dd307272505edcf18f8fbcbd91f18e1a9683af02804ee42fd289c`

The public Google Drive file IDs linked by the official README are:

- GT: `1KjRZcyA1THRzHZhX2yTGMtOdUW_wuGsI`
- LQ: `1wOaquAjnuzCh6Jv3CJz76mgnx4nfZgBY`

Downloaded archive evidence:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| GT.zip | 5,844,293,121 | `449fd43f27df42a31bfee2eef60cf09788315f2b0801ffbbf42b8f68420b0e77` |
| LQ.zip | 2,444,345,354 | `b61ac0a23501b0ee968b463bb07fa179615193138d6fe2c003a22fb716b11446` |

Both archives passed complete CRC checks before extraction. Each contains 20
category directories and 1,500 images, for 1,520 ZIP members total.

## Why `14Lowlight` is the contract benchmark

The main paper's Table 1 reports isolated and coupled degradations in separate
columns. Its isolated columns are Blur, Noise, JPEG, Haze, Rain, Raindrop, and
Lowlight; coupled cases such as `L+H`, `L+B`, `L+N`, and `L+J` have their own
columns. The FoundIR row reports `18.98 dB` for isolated Lowlight, matching the
contract's second-stage `18.9 dB` Lowlight target. The other contract task
targets likewise correspond to the isolated Table 1 columns. They do not
describe an aggregate of all categories containing Lowlight.

The supplement's Table 1 independently specifies exactly 50 Lowlight test
pairs. The official metadata maps those pairs to `14Lowlight`, IDs
`1101–1150`. Therefore the independent Lowlight Phase 1 gate is frozen to
those 50 pairs.

The README's optional specialist routing ranges cover multiple isolated and
coupled categories. Those ranges describe which outputs receive an optional
illumination-specialist refinement; they do not redefine the isolated
Lowlight benchmark.

## Frozen metric protocol

For each of the 50 manifest pairs:

1. use the official LQ image as the restoration input;
2. restore to the exact original image dimensions;
3. compare against the corresponding official GT image by manifest identity;
4. load both result and GT through PIL and convert to RGB;
5. convert to contiguous float32 NCHW tensors in `[0,1]` by dividing by 255;
6. calculate PSNR, SSIM, and LPIPS using the released FoundIR `pyiqa` calls;
7. apply no metric-time resizing, cropping, border shaving, Y-channel
   conversion, or per-image weighting;
8. report the arithmetic mean over all 50 pairs, plus per-image evidence.

The paper explicitly states that PSNR and SSIM are calculated on RGB. Its
model inference uses 1024-by-1024 patching and stitching for large images, but
the released metric script evaluates the resulting full-resolution images.
The official repository does not include the `environment.yml` referenced by
its README and does not pin a `pyiqa` version or metric-weight hashes. Before
the baseline/Qwen evaluation, this project must therefore freeze and record
the exact compatible `pyiqa` version, LPIPS backbone/weight hash, and runtime
environment, then use the same frozen evaluator for both systems.

Phase 1 pass thresholds remain:

- mean PSNR at least `15.1 dB`;
- mean LPIPS at most `0.34`.

SSIM is reported as supporting evidence but is not a contract pass threshold.

## Validation results

The fail-closed validator checked all 1,500 official pairs, not only the target
subset:

- exact 20-category composition against official metadata;
- exact category and pair-ID coverage (the archive normalizes some metadata
  filename extensions to PNG);
- exactly one GT and one LQ file per metadata pair;
- full image decoding;
- positive dimensions and exact GT/LQ dimension equality;
- SHA-256 and byte size for every GT and LQ image;
- exact SHA-256 overlap against the 2,000-source training manifest.

All 1,500 pairs passed structure, readability, and dimension validation. The
50 isolated Lowlight pairs are all RGB and have these resolutions:

| Resolution | Pairs |
|---|---:|
| 2720 x 1528 | 30 |
| 3008 x 1688 | 16 |
| 3680 x 2456 | 4 |

There is zero exact SHA-256 overlap between the 50 isolated Lowlight GT images
and the 2,000 training sources.

The broader 1,500-pair release does contain 50 exact matches to training
sources: all are category `17Lowlight_JPEG`, IDs `1301–1350` (46 DIV2K and 4
LSDIR sources). This does not contaminate the selected isolated Lowlight gate,
but it is retained in the JSON audit and must be addressed before any future
claim involving that coupled category.

## Reproducible artifacts

- acquisition/extraction launcher: `smoke/run_foundir_official_validation.sh`
- safe extractor: `smoke/extract_foundir_archive.py`
- validator: `smoke/validate_foundir_official.py`
- isolated Lowlight CSV: `manifests/foundir_lowlight_official.csv`
- isolated Lowlight JSON and overlap report:
  `manifests/foundir_lowlight_official.json`
- all-FoundIR GT hash index: `manifests/foundir_test_gt_sha256.json`
- focused unit tests: `tests/test_foundir_acquisition.py`

Manifest hashes:

- CSV: `5ea587a5da65902687c5f6d78b4c1ed4b80e2a3e0ecdaab5147979e003603a0f`
- JSON: `a5f5a536866cfa8e3980c5b6e50dea88616d0fce46c0a6b1027c4f95929e5bbc`
- GT hash index:
  `3b4b2717439e414d7aeb48729d46fa258fc0b8efa3ea76c2f7d5f78854238dd3`

The archives, extracted images, paper PDFs, and supplement PDF remain ignored
runtime assets under `downloads/`. Tracked manifests contain the provenance
needed to identify them without placing large data in Git.

## Next authorized work

Proceed to selecting and freezing a real vision-capable Qwen-VL checkpoint,
then implement the fair internal baseline/Qwen/oracle comparison. Do not run
either system on these official images until both system definitions and the
shared evaluator are frozen.
