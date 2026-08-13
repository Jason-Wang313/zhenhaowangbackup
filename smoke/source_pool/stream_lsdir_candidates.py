#!/usr/bin/env python3
"""Stream official LSDIR shard-00 and stop after enough train candidates."""
from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

import requests
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[2]
URL = "https://huggingface.co/ofsoundof/LSDIR/resolve/main/shard-00.tar.gz"
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1300)
    parser.add_argument("--output", type=Path, default=ROOT / "data/clean_pool/lsdir")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the authorized shard stream from stdin (recommended with curl).",
    )
    parser.add_argument("--archive", type=Path, help="Read an already downloaded official shard archive.")
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("BAD_COUNT")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.stdin and args.archive is not None:
        raise SystemExit("CHOOSE_ONLY_ONE_INPUT")
    if args.archive is not None:
        if not args.archive.is_file():
            raise SystemExit(f"MISSING_ARCHIVE {args.archive}")
        stream_context = args.archive.open("rb")
    elif args.stdin:
        stream_context = contextlib.nullcontext(sys.stdin.buffer)
    else:
        token = get_token()
        if not token:
            raise SystemExit("HF_NOT_AUTHENTICATED: run `hf auth login`")
        response = requests.get(
            URL,
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            timeout=(30, 300),
        )
        response.raise_for_status()
        response.raw.decode_content = True
        stream_context = response

    with stream_context as stream_owner:
        stream = stream_owner if (args.stdin or args.archive is not None) else stream_owner.raw
        selected = 0
        with tarfile.open(fileobj=stream, mode="r|gz", bufsize=4 * 1024 * 1024) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                member_path = PurePosixPath(member.name)
                if member_path.suffix.lower() not in EXTENSIONS:
                    continue
                # shard-00 is an official HR training shard. Reject anything
                # whose path explicitly identifies validation/test/LR content.
                lowered = "/".join(member_path.parts).lower()
                if any(marker in lowered for marker in ("/val/", "/test/", "/x2/", "/x3/", "/x4/")):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                destination = args.output / member_path.name
                if destination.exists():
                    if destination.stat().st_size == 0:
                        raise SystemExit(f"EMPTY_EXISTING_DESTINATION {destination}")
                    selected += 1
                    if selected == args.count:
                        print(f"LSDIR_STREAM_COMPLETE {selected} candidates", flush=True)
                        return
                    continue
                temporary = destination.with_name(f".{destination.name}.partial")
                temporary.unlink(missing_ok=True)
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle, length=1024 * 1024)
                temporary.replace(destination)
                selected += 1
                if selected % 100 == 0:
                    print(f"LSDIR_STREAM_PROGRESS {selected}/{args.count}", flush=True)
                if selected == args.count:
                    print(f"LSDIR_STREAM_COMPLETE {selected} candidates", flush=True)
                    return
    raise SystemExit(f"LSDIR_STREAM_INCOMPLETE selected={selected} expected={args.count}")


if __name__ == "__main__":
    main()
