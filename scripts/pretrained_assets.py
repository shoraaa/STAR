#!/usr/bin/env python3
"""Manage pretrained model assets that are too large for normal Git.

The script operates on checkpoint/model files under survey/NRS and supports:

  manifest  write a CSV manifest with paths, sizes, and SHA-256 hashes
  verify    check that files from a manifest exist and match size/hash
  pack      create a tar.gz archive containing all manifest entries
  unpack    extract such an archive into a checkout
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tarfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = ROOT / "survey" / "NRS"
DEFAULT_MANIFEST = ROOT / "pretrained_assets_manifest.csv"
DEFAULT_ARCHIVE = ROOT / "pretrained-assets.tar.gz"
ASSET_SUFFIXES = (".pt", ".pkl", ".best", ".pth")
MANIFEST_FIELDS = ("path", "size_bytes", "sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage pretrained model assets for this repo.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest CSV path; default: pretrained_assets_manifest.csv",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="write a manifest for local assets")
    manifest.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    manifest.add_argument(
        "--no-hash",
        action="store_true",
        help="record file sizes only; faster but less useful for integrity checks",
    )

    verify = subparsers.add_parser("verify", help="verify assets listed in a manifest")
    verify.add_argument("--allow-missing-hash", action="store_true")

    pack = subparsers.add_parser("pack", help="create a tar.gz archive from manifest entries")
    pack.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    pack.add_argument("--verify-first", action="store_true")

    unpack = subparsers.add_parser("unpack", help="extract an archive into this checkout")
    unpack.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    unpack.add_argument("--verify-after", action="store_true")

    return parser.parse_args()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_paths(asset_root: Path) -> list[Path]:
    root = asset_root if asset_root.is_absolute() else ROOT / asset_root
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in ASSET_SUFFIXES)


def write_manifest(manifest_path: Path, paths: Iterable[Path], *, include_hash: bool) -> int:
    manifest_path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        rows.append(
            {
                "path": repo_relative(path),
                "size_bytes": str(path.stat().st_size),
                "sha256": sha256_file(path) if include_hash else "",
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    total_size = sum(int(row["size_bytes"]) for row in rows)
    print(f"Wrote {len(rows)} assets to {repo_relative(manifest_path)} ({total_size:,} bytes).")
    return len(rows)


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    manifest_path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_manifest(manifest_path: Path, *, allow_missing_hash: bool) -> bool:
    rows = read_manifest(manifest_path)
    ok = True
    for row in rows:
        rel_path = row["path"]
        path = ROOT / rel_path
        if not path.exists():
            print(f"missing: {rel_path}")
            ok = False
            continue
        expected_size = int(row["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            print(f"size mismatch: {rel_path} expected={expected_size} actual={actual_size}")
            ok = False
        expected_hash = row.get("sha256", "")
        if expected_hash:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                print(f"sha256 mismatch: {rel_path}")
                ok = False
        elif not allow_missing_hash:
            print(f"missing sha256 in manifest: {rel_path}")
            ok = False
    print(f"Verified {len(rows)} manifest entries." if ok else "Asset verification failed.")
    return ok


def pack_assets(manifest_path: Path, archive_path: Path) -> None:
    rows = read_manifest(manifest_path)
    archive_path = archive_path if archive_path.is_absolute() else ROOT / archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for row in rows:
            rel_path = row["path"]
            archive.add(ROOT / rel_path, arcname=rel_path)
    print(f"Packed {len(rows)} assets into {repo_relative(archive_path)}.")


def unpack_assets(archive_path: Path) -> None:
    archive_path = archive_path if archive_path.is_absolute() else ROOT / archive_path
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(ROOT)
    print(f"Unpacked {repo_relative(archive_path)}.")


def main() -> int:
    args = parse_args()
    if args.command == "manifest":
        write_manifest(args.manifest, asset_paths(args.asset_root), include_hash=not args.no_hash)
        return 0
    if args.command == "verify":
        return 0 if verify_manifest(args.manifest, allow_missing_hash=args.allow_missing_hash) else 1
    if args.command == "pack":
        if args.verify_first and not verify_manifest(args.manifest, allow_missing_hash=False):
            return 1
        pack_assets(args.manifest, args.archive)
        return 0
    if args.command == "unpack":
        unpack_assets(args.archive)
        if args.verify_after:
            return 0 if verify_manifest(args.manifest, allow_missing_hash=False) else 1
        return 0
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
