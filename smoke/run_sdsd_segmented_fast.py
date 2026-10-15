#!/usr/bin/env python3
"""Fast bounded-range parameters for the immutable SDSD segment downloader."""

from __future__ import annotations

import smoke.acquire_sdsd_development_archives_segmented as implementation


implementation.SEGMENT_BYTES = 8 * 1024 * 1024
implementation.PARALLEL_REQUESTS = 32


if __name__ == "__main__":
    raise SystemExit(implementation.main())
