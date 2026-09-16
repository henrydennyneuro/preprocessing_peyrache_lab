"""
truncate_session.py
===================
Repair utility for Intan sessions where a recording was interrupted mid-capture
(e.g. system crash, out-of-memory).  In that situation Intan's write buffers can
flush more samples to disk than the timestamp counter (time.dat) recorded, leaving
orphaned bytes at the tail of one or more .dat files.

This script uses time.dat as the authoritative sample count and truncates any
larger signal file to the matching byte boundary.  The truncation is in-place and
O(1) — no data is recopied, the OS simply updates the file length.

Usage
-----
    # Preview what would change (no writes):
    python truncate_session.py <session_folder> --dry-run

    # Apply truncation to a single session folder:
    python truncate_session.py <session_folder>

    # Process all session folders under a recording-day directory:
    python truncate_session.py <data_dir> --animal-id <ID> [--dry-run]

    # Also check auxiliary.dat (see note below before using):
    python truncate_session.py <session_folder> --include-auxiliary

Notes
-----
auxiliary.dat is excluded by default.  In a normal crash scenario it has a small
tail surplus identical to amplifier.dat.  However, if the auxiliary recording was
disrupted differently (e.g. large size mismatch with no clean frame boundary), the
floor-division frame estimate will be wrong and truncation could corrupt the file.
Use --include-auxiliary only after verifying that the surplus is small and that the
computed bytes/frame is a clean multiple of 2.
"""

import argparse
import re
import sys
from pathlib import Path


SIGNAL_FILES_DEFAULT = [
    "amplifier.dat",
    "analogin.dat",
    "digitalin.dat",
]

SIGNAL_FILES_WITH_AUX = [
    "amplifier.dat",
    "auxiliary.dat",
    "analogin.dat",
    "digitalin.dat",
]


def _n_samples_from_time_dat(session_dir: Path) -> int:
    time_dat = session_dir / "time.dat"
    if not time_dat.exists():
        raise FileNotFoundError(f"time.dat not found in {session_dir}")
    size = time_dat.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"time.dat size ({size} bytes) is not a multiple of 4")
    return size // 4


def _check_session(session_dir: Path, dry_run: bool, include_auxiliary: bool = False) -> dict:
    """
    Inspect all signal files in session_dir against time.dat.
    Truncates files in-place unless dry_run is True.
    Returns a summary dict.
    """
    n_samples = _n_samples_from_time_dat(session_dir)
    checks = []
    signal_files = SIGNAL_FILES_WITH_AUX if include_auxiliary else SIGNAL_FILES_DEFAULT

    for fname in signal_files:
        fpath = session_dir / fname
        if not fpath.exists():
            checks.append({"file": fname, "status": "missing"})
            continue

        size = fpath.stat().st_size
        if size == 0:
            checks.append({"file": fname, "status": "empty"})
            continue

        bytes_per_frame = size // n_samples
        if bytes_per_frame == 0:
            checks.append({"file": fname, "status": "too_small", "size": size})
            continue

        expected = n_samples * bytes_per_frame
        surplus  = size - expected

        if surplus == 0:
            checks.append({"file": fname, "status": "ok", "size": size})
        else:
            if not dry_run:
                with open(fpath, "r+b") as f:
                    f.truncate(expected)
            checks.append({
                "file":            fname,
                "status":          "would_truncate" if dry_run else "truncated",
                "size":            size,
                "expected":        expected,
                "surplus_bytes":   surplus,
                "surplus_samples": surplus // bytes_per_frame,
                "bytes_per_frame": bytes_per_frame,
            })

    return {"n_samples": n_samples, "checks": checks}


def _print_results(session_dir: Path, results: dict, dry_run: bool) -> None:
    n = results["n_samples"]
    print(f"\n  {session_dir.name}  ({n:,} samples from time.dat)")
    any_action = False
    for c in results["checks"]:
        st    = c["status"]
        fname = c["file"]
        if st == "ok":
            print(f"    {fname:20s}  OK")
        elif st in ("missing", "empty", "too_small"):
            print(f"    {fname:20s}  skipped ({st})")
        elif st in ("would_truncate", "truncated"):
            any_action = True
            verb = "Would truncate" if dry_run else "Truncated"
            print(
                f"    {fname:20s}  {verb}: -{c['surplus_bytes']:,} bytes "
                f"({c['surplus_samples']:,} samples, {c['bytes_per_frame']} bytes/frame)"
            )
    if not any_action:
        print("    All files aligned.")


def _find_session_folders(data_dir: Path, animal_id: str) -> list:
    pattern = re.compile(r"^" + re.escape(animal_id) + r"_\d{6}_\d{6}$")
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and pattern.match(d.name))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Truncate Intan .dat signal files to match time.dat sample count."
    )
    ap.add_argument(
        "path", type=Path,
        help="Session folder (single) or parent data directory (with --animal-id)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report mismatches without writing any files",
    )
    ap.add_argument(
        "--animal-id", default=None,
        help="Animal ID prefix used to locate session folders under PATH",
    )
    ap.add_argument(
        "--include-auxiliary", action="store_true",
        help="Also check auxiliary.dat (see script docstring before using)",
    )
    args = ap.parse_args()

    if args.animal_id:
        sessions = _find_session_folders(args.path, args.animal_id)
        if not sessions:
            print(f"No session folders found under {args.path} for animal '{args.animal_id}'")
            sys.exit(1)
        print(f"Found {len(sessions)} session folder(s) under {args.path}")
    else:
        sessions = [args.path]

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{prefix}Checking session(s)...")

    errors = 0
    for sdir in sessions:
        try:
            results = _check_session(sdir, dry_run=args.dry_run,
                                     include_auxiliary=args.include_auxiliary)
            _print_results(sdir, results, dry_run=args.dry_run)
        except Exception as exc:
            print(f"\n  {sdir.name}  ERROR: {exc}")
            errors += 1

    print()
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
