"""
Batch version of the check_waveform_files pipeline step (method and verdict:
preprocessing_pipeline/waveform_check.py).

  python CheckWaveformFiles.py <session_list.txt> <out_dir>
      checks every session; writes <out_dir>/units_<session>.csv,
      arrays_<session>.npz (averaged waveforms, reused on re-runs),
      waveform_check_units.csv and waveform_check_sessions.csv; modifies
      nothing in the session folders.
  python CheckWaveformFiles.py --write <session_list.txt> <out_dir>
      saves those results into each session folder (waveform_check.save_session).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing_pipeline import waveform_check  # noqa: E402


def write_session_files(sessions, out):
    summary = pd.read_csv(os.path.join(out, "waveform_check_sessions.csv")).set_index("session")
    for s in sessions:
        b = os.path.basename(os.path.normpath(s))
        units_csv = os.path.join(out, f"units_{b}.csv")
        if b not in summary.index or not os.path.exists(units_csv) or os.path.getsize(units_csv) < 10:
            print(f"{b}: no check results; nothing written")
            continue
        ids = pd.read_csv(os.path.join(s, f"{b}_unit_ids.csv"))
        verdict = summary.loc[b, "verdict"]
        path = waveform_check.save_session(s, pd.read_csv(units_csv), verdict, ids)
        print(f"{b}: {verdict}, {len(ids)} units -> {path}")


def main():
    if sys.argv[1] == "--write":
        write_session_files([l.strip() for l in open(sys.argv[2]) if l.strip()], sys.argv[3])
        return
    sessions = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    unit_tables, summary = [], []
    for s in sessions:
        b = os.path.basename(s)
        try:
            t, issues, notes, shift_ok = waveform_check.check_session(s, cache=os.path.join(out, f"arrays_{b}.npz"))
        except Exception as e:                                    # report and move on
            t, issues, notes, shift_ok = pd.DataFrame(), [f"{type(e).__name__}: {e}"], [], {}
        t.to_csv(os.path.join(out, f"units_{b}.csv"), index=False)
        unit_tables.append(t)
        summary.append(waveform_check.summarize(b, t, issues, notes, shift_ok))
        print(summary[-1], flush=True)
    pd.concat(unit_tables, ignore_index=True).to_csv(os.path.join(out, "waveform_check_units.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out, "waveform_check_sessions.csv"), index=False)


if __name__ == "__main__":
    main()
