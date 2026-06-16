#!/usr/bin/env python3
"""Convert second-based durations in a TeX file to minutes or hours."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SECONDS_RE = re.compile(r"(?<![A-Za-z])(?P<value>\d+(?:\.\d+)?)s\b")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"

    hours = minutes / 60
    return f"{hours:.2f}h"


def convert_text(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        original = match.group(0)
        replacement = format_duration(float(match.group("value")))
        if replacement != original:
            count += 1
        return replacement

    return SECONDS_RE.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert durations like 120.50s to 2.01m, and to hours when the "
            "converted minute value is at least 60."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="paper.tex",
        type=Path,
        help="TeX file to update, default: paper.tex",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many replacements would be made without writing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if the file contains convertible seconds.",
    )
    args = parser.parse_args()

    text = args.path.read_text(encoding="utf-8")
    converted, count = convert_text(text)

    if args.check:
        if count:
            print(f"{args.path}: {count} duration token(s) need conversion")
            return 1
        print(f"{args.path}: no duration conversions needed")
        return 0

    if args.dry_run:
        print(f"{args.path}: would convert {count} duration token(s)")
        return 0

    if converted != text:
        args.path.write_text(converted, encoding="utf-8")
    print(f"{args.path}: converted {count} duration token(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
