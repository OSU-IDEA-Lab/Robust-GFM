#!/usr/bin/env python3
"""Hardlink-dedupe compiled Freebase block-conditional channels across variants.

Every variant must compile to the identical semantic message program, so the
per-channel CSR files are expected to be bitwise identical to the reference
variant's. Where the manifest csr_sha256 values match, this replaces the
duplicate indptr.npy/indices.npy with hardlinks to the reference copy,
reclaiming ~16G per variant on the quota-limited share. A mismatched channel is
an invariance violation and aborts with a nonzero exit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_manifest(root: Path) -> dict[int, dict]:
    payload = json.loads((root / "model_input_hashes.json").read_text())
    return {int(c["channel_id"]): c for c in payload["channels"]}


def channel_dirs(root: Path) -> dict[int, Path]:
    out = {}
    for p in sorted((root / "channels").iterdir()):
        if p.is_dir():
            out[int(p.name.split("_", 1)[0])] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    args = ap.parse_args()

    ref_manifest = load_manifest(args.reference)
    tgt_manifest = load_manifest(args.target)
    ref_dirs = channel_dirs(args.reference)
    tgt_dirs = channel_dirs(args.target)

    if set(ref_manifest) != set(tgt_manifest):
        raise SystemExit(
            f"Channel id sets differ: {sorted(set(ref_manifest) ^ set(tgt_manifest))[:10]}"
        )

    mismatched = [
        cid for cid in sorted(ref_manifest)
        if ref_manifest[cid]["csr_sha256"] != tgt_manifest[cid]["csr_sha256"]
    ]
    if mismatched:
        for cid in mismatched[:10]:
            print(
                f"MISMATCH channel {cid} {ref_manifest[cid]['semantic_relation']}: "
                f"{ref_manifest[cid]['csr_sha256'][:12]} != {tgt_manifest[cid]['csr_sha256'][:12]}"
            )
        raise SystemExit(
            f"{len(mismatched)} channels differ from reference; invariance violated - not deduping."
        )

    saved = 0
    linked = 0
    for cid in sorted(ref_manifest):
        rdir, tdir = ref_dirs[cid], tgt_dirs[cid]
        for name in ("indptr.npy", "indices.npy"):
            rfile, tfile = rdir / name, tdir / name
            if rfile.stat().st_ino == tfile.stat().st_ino:
                continue  # already hardlinked
            saved += tfile.stat().st_size
            tmp = tfile.with_suffix(".tmp_link")
            os.link(rfile, tmp)
            os.replace(tmp, tfile)
            linked += 1
    print(f"Deduped {linked} files, reclaimed {saved / 2**30:.2f} GiB "
          f"({args.target} -> hardlinks into {args.reference})")


if __name__ == "__main__":
    main()
