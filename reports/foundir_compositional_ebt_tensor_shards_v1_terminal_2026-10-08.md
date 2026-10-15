# FoundIR compositional EBT tensor shards v1 — terminal result

**READY.** The one-shot training I/O bank is frozen and verified.

- 60,000 lower-order train records; zero validation/triple records materialized.
- Eight deterministic 128x128 crop/augmentation views per record.
- 480,000 paired degraded/target samples.
- 469 contiguous structured NumPy shards; approximately 44 GiB on disk.
- Full shard SHA-256 verification passed, followed by independent structural check.
- Build wall time: `347.605973 seconds` (`0.096557 hours`).
- GPU usage: `0 GPU-hours`.
- Official FoundIR/sealed/triple access: none.

Identities:

- Config SHA-256: `16da0a5fee43c47dab5ef6770b43b54e3ba761257c9620fdccd6cf535820119a`
- Builder SHA-256: `f69e922aa03fdeeda1b4ec5e63ee45e5e799daa08c888bd27781f660b99c14c9`
- Manifest SHA-256: `561037670c88ead7a115687ef19da51eed061c329fd53b8e9b383b598fb5f3ff`
- Metadata SHA-256: `a70a54be261548e3b1fa87875006737e5956ca2557d9cb2b8fb4353d9ae1dd51`
