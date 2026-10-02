"""
Carry a session's waveform files over to a rebuilt NWB when the raw .dat is gone.

_mean_wf.csv, _max_ch.csv and _waveform_parameters.csv in the session folder are
still indexed by the OLD NWB's unit numbers. Each new unit (from
{basename}_unit_ids.csv, so run extract_unit_ids on the new NWB first) is matched
to the old unit with the IDENTICAL spike train (same spike_hash, computed from
the old NWB the same way); its rows are copied under the new unit number. Old
units with no match (e.g. clusters rejected in a re-curation) are dropped; a new
unit with no match is an error (its waveform needs the .dat). The channel layout
(electrode groups) of the two NWBs must be identical. Old files are backed up
first; the new ones get uid + spike_hash columns (_max_ch, _waveform_parameters)
and provenance stamps. Then run check_waveform_files.

Usage: python CarryOverWaveforms.py <session_dir> <old_nwb_path> <backup_dir>
"""
import os
import shutil
import sys

import h5py
import numpy as np
import pandas as pd
import pynapple as nap

from preprocessing_pipeline import unit_ids, waveform_check


def nwb_hashes(nwb_path):
    with h5py.File(nwb_path, "r") as h:
        ends = h["units/spike_times_index"][()]
        trains = np.split(h["units/spike_times"][()], ends[:-1])
        ids = h["units/id"][()]
    return {int(u): unit_ids.spike_hash(np.rint(t * unit_ids.FS).astype(np.int64)) for u, t in zip(ids, trains)}


def electrode_groups(nwb_path):
    with h5py.File(nwb_path, "r") as h:
        e = h["general/extracellular_ephys/electrodes"]
        names = [x.decode() if isinstance(x, bytes) else x for x in e["group_name"][()]]
        return list(zip(e["id"][()].tolist(), [int(n.split("_")[0][5:]) for n in names]))


def main():
    session_dir, old_nwb, backup_dir = sys.argv[1:4]
    b = os.path.basename(os.path.normpath(session_dir))
    new_nwb = unit_ids.session_nwb(session_dir)
    if electrode_groups(old_nwb) != electrode_groups(new_nwb):
        raise RuntimeError("Electrode groups differ between the old and new NWB; waveform columns would not line up")
    ids = unit_ids.load_verified(session_dir, nap.load_file(new_nwb)["units"])
    old_by_hash = {h: u for u, h in nwb_hashes(old_nwb).items()}
    mapping = {int(r.unit): old_by_hash.get(r.spike_hash) for r in ids.itertuples()}
    missing = [u for u, o in mapping.items() if o is None]
    if missing:
        raise RuntimeError(f"New units {missing} have no identical spike train in the old NWB; they need the .dat")
    dropped = sorted(set(old_by_hash.values()) - set(mapping.values()))

    os.makedirs(backup_dir, exist_ok=True)
    p = lambda s: os.path.join(session_dir, f"{b}{s}")
    for s in waveform_check.WAVEFORM_FILES:
        shutil.copy2(p(s), os.path.join(backup_dir, f"{b}{s}"))

    wf = pd.read_csv(p("_mean_wf.csv"), index_col=[0, 1])
    max_ch = pd.read_csv(p("_max_ch.csv"), index_col=0)["max_channel"]
    t2p = pd.read_csv(p("_waveform_parameters.csv"), index_col=0)["trough_to_peak"]
    new_units = sorted(mapping)
    parts = []
    for u in new_units:
        block = wf.loc[[mapping[u]]].copy()
        block.index = pd.MultiIndex.from_arrays([[u] * len(block), block.index.get_level_values(1)],
                                                names=wf.index.names)
        parts.append(block)
    pd.concat(parts).to_csv(p("_mean_wf.csv"))
    pd.Series({u: max_ch[mapping[u]] for u in new_units}, name="max_channel").to_csv(p("_max_ch.csv"))
    pd.Series({u: t2p[mapping[u]] for u in new_units}, name="trough_to_peak").to_csv(p("_waveform_parameters.csv"))
    waveform_check.tag_parameter_files(session_dir, ids)
    for s in waveform_check.WAVEFORM_FILES:
        unit_ids.record_output(session_dir, f"{b}{s}", ids,
                               "extract_waveform_parameters (old NWB); carried over by uid + spike_hash")
    moved = sum(1 for u in new_units if mapping[u] != u)
    print(f"{b}: {len(new_units)} units carried over ({moved} renumbered), {len(dropped)} old units dropped "
          f"(old ids {dropped}); old files backed up to {backup_dir}")


if __name__ == "__main__":
    main()
