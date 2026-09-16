"""
run_lfp_sleep.py
================
Standalone entry point for LFP downsampling and automated sleep scoring.
Runs Process_LFPfromDat and/or SleepScoreMaster via MATLAB -batch on an
already-preprocessed recording folder.

USAGE
-----
    python run_lfp_sleep.py D:/data/B3218-260420

Both steps (LFP + sleep scoring) run by default.

OPTIONS
-------
  --out-fs HZ           LFP output sample rate in Hz (default: 1250)
  --lo-pass HZ          Lowpass cutoff before downsampling in Hz (default: 450)
  --skip-lfp            Skip Process_LFPfromDat; re-use existing .eeg file
  --skip-sleep-score    Skip SleepScoreMaster only
  --dry-run             Print MATLAB command without executing it
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pipeline.matlab_runner import run_lfp_sleep
from pipeline import config as cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="LFP downsampling + sleep scoring via MATLAB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "base_path", type=Path,
        help="Recording folder containing <name>.dat and <name>.xml"
    )
    p.add_argument(
        "--out-fs", type=int, default=cfg.LFP_SAMPLE_RATE, metavar="HZ",
        help=f"LFP output sample rate in Hz (default: {cfg.LFP_SAMPLE_RATE})"
    )
    p.add_argument(
        "--lo-pass", type=int, default=cfg.LFP_LO_PASS, metavar="HZ",
        help=f"Lowpass cutoff in Hz (default: {cfg.LFP_LO_PASS})"
    )
    p.add_argument(
        "--skip-lfp", action="store_true",
        help="Skip Process_LFPfromDat; re-use an existing .eeg file"
    )
    p.add_argument(
        "--skip-sleep-score", action="store_true",
        help="Skip SleepScoreMaster"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print MATLAB command without executing it"
    )
    return p.parse_args()


def main():
    args = parse_args()

    base_path = args.base_path.resolve()
    if not base_path.is_dir():
        sys.exit(f"ERROR: directory does not exist: {base_path}")

    print(f"\n{'='*60}")
    print(f"  LFP + Sleep Scoring")
    print(f"  Recording  : {base_path}")
    print(f"  LFP out fs : {args.out_fs} Hz  (lopass {args.lo_pass} Hz)")
    print(f"  LFP step   : {'SKIPPED' if args.skip_lfp else 'Process_LFPfromDat'}")
    print(f"  Sleep step : {'SKIPPED' if args.skip_sleep_score else 'SleepScoreMaster'}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'='*60}\n")

    run_lfp_sleep(
        base_path        = base_path,
        out_fs           = args.out_fs,
        lo_pass          = args.lo_pass,
        skip_lfp         = args.skip_lfp,
        skip_sleep_score = args.skip_sleep_score,
        dry_run          = args.dry_run,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
