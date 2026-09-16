"""
batch_process.py
================
Runs master_preprocessing.py on every recording folder found inside a parent
directory.  Accepts the same arguments as master_preprocessing.py; --data-dir
is re-interpreted as the parent directory containing multiple per-recording
subfolders rather than a single recording folder.

A subfolder is treated as a recording if it contains at least one Intan session
folder whose name matches the pattern <AnimalID>_<YYMMDD>_<HHMMSS>.

If one recording fails, a warning is printed and processing continues with the
next recording.  A summary of successes and failures is printed at the end.

USAGE
-----
    python batch_process.py --data-dir PATH [same options as master_preprocessing.py]

EXAMPLES
--------
    # Process all recordings under Data/B3218/, auto-detect probe and animal ID
    python batch_process.py --data-dir "C:/Data/B3218/"

    # With a specific probe
    python batch_process.py --data-dir "C:/Data/B3218/" --probe cambridge_p64

    # Two-probe recordings, skip sorting
    python batch_process.py --data-dir "C:/Data/B3218/" \\
        --probe cambridge_p64 neuronexus_buzsaki64 --skip-sorting

    # Dry run to preview what would be processed
    python batch_process.py --data-dir "C:/Data/B3218/" --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR     = Path(__file__).resolve().parent
_MASTER         = _SCRIPT_DIR / "master_preprocessing.py"
_SORTER         = _SCRIPT_DIR.parent / "SpikeSorter" / "run_spike_sorting.py"
_SESSION_RE     = re.compile(r"^\w+_\d{6}_\d{6}$")


def _is_recording_folder(path: Path) -> bool:
    """True if path contains at least one Intan session subfolder."""
    try:
        return any(
            _SESSION_RE.match(p.name)
            for p in path.iterdir()
            if p.is_dir()
        )
    except PermissionError:
        return False


def _find_recordings(parent: Path) -> list:
    """Return sorted list of recording subfolders inside parent."""
    return sorted(
        p for p in parent.iterdir()
        if p.is_dir() and _is_recording_folder(p)
    )


def _find_export_recordings(parent: Path) -> list:
    """Return sorted list of subfolders that contain a kilosort4/ directory."""
    try:
        return sorted(
            p for p in parent.iterdir()
            if p.is_dir() and (p / "kilosort4").is_dir()
        )
    except PermissionError:
        return []


def _build_export_command(recording_dir: Path, args: argparse.Namespace) -> list:
    """Construct the run_spike_sorting.py --export-only command for one recording."""
    merge_name = recording_dir.name
    cmd = [sys.executable, str(_SORTER), merge_name,
           "--data-dir", str(recording_dir),
           "--export-only"]
    if args.probe:
        cmd += ["--probe"] + args.probe
    if args.output_format != "neurosuite":
        cmd += ["--output-format", args.output_format]
    if args.keep_labels:
        cmd += ["--keep-labels"] + args.keep_labels
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_command(recording_dir: Path, args: argparse.Namespace) -> list:
    """Construct the master_preprocessing.py command for one recording."""
    cmd = [sys.executable, str(_MASTER)]

    if args.animal_id:
        cmd.append(args.animal_id)

    cmd += ["--data-dir", str(recording_dir)]

    if args.probe:
        cmd += ["--probe"] + args.probe
    if args.layout != "staggered":
        cmd += ["--layout", args.layout]
    if args.site_spacing != 20.0:
        cmd += ["--site-spacing", str(args.site_spacing)]
    if args.output_format != "neurosuite":
        cmd += ["--output-format", args.output_format]
    if args.skip_sorting:
        cmd.append("--skip-sorting")
    if args.skip_sleep_score:
        cmd.append("--skip-sleep-score")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.sample_rate is not None:
        cmd += ["--sample-rate", str(args.sample_rate)]
    if args.no_analogin:
        cmd.append("--no-analogin")
    if args.no_digitalin:
        cmd.append("--no-digitalin")
    if args.no_auxiliary:
        cmd.append("--no-auxiliary")
    if args.work_dir:
        cmd += ["--work-dir", str(args.work_dir)]
    if args.keep_work_dir:
        cmd.append("--keep-work-dir")

    return cmd


def _parse_args():
    p = argparse.ArgumentParser(
        description="Batch wrapper: runs master_preprocessing.py over multiple recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "animal_id", nargs="?", default=None,
        help="AnimalID prefix (e.g. B3218). Auto-detected per recording if omitted.",
    )
    p.add_argument(
        "--data-dir", type=Path, required=True,
        help="Parent directory containing per-recording subfolders.",
    )
    p.add_argument("--probe", nargs="+", default=None, metavar="NAME_OR_PATH",
        help="Probe name(s) or path(s). Pass two values for two-probe recordings.")
    p.add_argument("--layout", choices=["staggered", "linear", "columns"],
        default="staggered")
    p.add_argument("--site-spacing", type=float, default=20.0, metavar="MICRONS")
    p.add_argument("--output-format", choices=["neurosuite", "phy"],
        default="neurosuite")
    p.add_argument("--skip-sorting", action="store_true",
        help="Pre-process only; do not launch spike sorter.")
    p.add_argument("--skip-sleep-score", action="store_true",
        help="Skip LFP downsampling and sleep scoring (steps 8-9).")
    p.add_argument("--dry-run", action="store_true",
        help="Print what would be done without doing it.")
    p.add_argument("--sample-rate", type=int, default=None, metavar="HZ",
        help="Recording sample rate. Auto-detected from info.rhd if omitted.")
    p.add_argument("--no-analogin",  action="store_true")
    p.add_argument("--no-digitalin", action="store_true")
    p.add_argument("--no-auxiliary", action="store_true")
    p.add_argument("--work-dir", type=Path, default=None,
        help="Fast scratch directory for processing.")
    p.add_argument("--keep-work-dir", action="store_true",
        help="Do not delete scratch copy after a successful run.")
    p.add_argument("--export-only", action="store_true",
        help="Skip preprocessing and KiloSort; re-export from existing "
             "kilosort4/ folders. Detects recordings by kilosort4/ presence "
             "rather than Intan session folders.")
    p.add_argument("--keep-labels", nargs="+", default=None, metavar="LABEL",
        help="Only export clusters with these Phy labels (e.g. --keep-labels good). "
             "Used with --export-only. Reads cluster_group.tsv from each kilosort4/ folder.")
    return p.parse_args()


def main():
    args    = _parse_args()
    parent  = args.data_dir.resolve()

    if not parent.is_dir():
        sys.exit(f"ERROR: --data-dir does not exist: {parent}")

    if args.export_only:
        recordings  = _find_export_recordings(parent)
        build_cmd   = _build_export_command
        mode_label  = "Batch Export (--export-only)"
        not_found   = (
            f"ERROR: no kilosort4/ folders found under {parent}\n"
            "  Each recording subfolder must contain a kilosort4/ directory."
        )
    else:
        recordings  = _find_recordings(parent)
        build_cmd   = _build_command
        mode_label  = "Batch SpikeSorter"
        not_found   = (
            f"ERROR: no recording folders found under {parent}\n"
            "  A recording folder must contain at least one Intan session\n"
            "  subfolder named <AnimalID>_<YYMMDD>_<HHMMSS>."
        )

    if not recordings:
        sys.exit(not_found)

    print(f"\n{'='*60}")
    print(f"  {mode_label}")
    print(f"  Parent directory  : {parent}")
    print(f"  Recordings found  : {len(recordings)}")
    for r in recordings:
        print(f"    {r.name}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[dry-run] Commands that would be executed:\n")
        for rec in recordings:
            print("  " + " ".join(build_cmd(rec, args)))
        return

    failed = []
    for i, rec in enumerate(recordings, 1):
        print(f"\n{'='*60}")
        print(f"  Recording {i}/{len(recordings)}: {rec.name}")
        print(f"{'='*60}")
        result = subprocess.run(build_cmd(rec, args))
        if result.returncode != 0:
            print(f"\n  WARNING: {rec.name} failed (exit code {result.returncode})"
                  " — continuing with next recording.")
            failed.append(rec.name)

    print(f"\n{'='*60}")
    print(f"  Batch complete.  "
          f"{len(recordings) - len(failed)}/{len(recordings)} succeeded.")
    if failed:
        print(f"  Failed recordings:")
        for name in failed:
            print(f"    {name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
