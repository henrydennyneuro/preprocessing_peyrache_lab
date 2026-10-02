"""
Relabel electrode groups (shanks) of a session NWB in place.

The label lives in three places, all variable-length strings, so they are
overwritten without resizing anything: units/location (per unit),
general/extracellular_ephys/electrodes/location (per channel) and the
'location' attribute of each electrode group (general/extracellular_ephys/
group{g}_...). Spike trains and everything else are untouched, so unit uids
and spike_hashes stay valid (re-run extract_unit_ids to refresh its location
column). Running it again with the same labels changes nothing. A rebuilt NWB
(nwbmatic GUI) takes its labels from the GUI, so enter the same labels there.

Usage:
    python RelabelNWBGroups.py <session_dir> <group>=<label> [<group>=<label> ...] [--dry-run]
    group = 0-based nwbmatic group (clu.N is group N-1)

261001 (histology): B3210 groups 2-8 and B3212 groups 5-8 are striatum:
    python RelabelNWBGroups.py D:/B3200/B3212/B3212-241008 5=STR 6=STR 7=STR 8=STR
"""
import argparse
import glob
import os

import h5py
import numpy as np
from pynwb import NWBHDF5IO

EPHYS = "general/extracellular_ephys"


def decode(x):
    return x.decode() if isinstance(x, bytes) else x


def group_of_name(name):
    """'group3_...' -> 3"""
    return int(decode(name).split("_")[0][len("group"):])


def current_labels(h):
    g = h[EPHYS]
    groups = {group_of_name(k): k for k in g.keys() if k.startswith("group")}
    units_group = h["units/group"][()]
    units_loc = [decode(x) for x in h["units/location"][()]]
    el_group = [group_of_name(x) for x in h[f"{EPHYS}/electrodes/group_name"][()]]
    el_loc = [decode(x) for x in h[f"{EPHYS}/electrodes/location"][()]]
    return groups, units_group, units_loc, el_group, el_loc


def main():
    parser = argparse.ArgumentParser(description="Relabel NWB electrode groups in place.")
    parser.add_argument("session_dir")
    parser.add_argument("labels", nargs="+", help="<group>=<label>, 0-based groups")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    new = {int(k): v for k, v in (x.split("=", 1) for x in args.labels)}

    nwb_files = glob.glob(os.path.join(args.session_dir, "pynapplenwb", "*.nwb"))
    if len(nwb_files) != 1:
        raise RuntimeError(f"Expected one NWB in {args.session_dir}/pynapplenwb, found {nwb_files}")
    nwb_path = nwb_files[0]

    with h5py.File(nwb_path, "r") as h:
        groups, units_group, units_loc, el_group, el_loc = current_labels(h)
        missing = sorted(set(new) - set(groups))
        if missing:
            raise RuntimeError(f"Groups {missing} not in the NWB (has {sorted(groups)})")
        print(nwb_path)
        for g in sorted(new):
            old_attr = decode(h[f"{EPHYS}/{groups[g]}"].attrs["location"])
            n_u = int(np.sum(units_group == g)); n_e = sum(x == g for x in el_group)
            print(f"  group {g}: '{old_attr}' -> '{new[g]}'  ({n_u} units, {n_e} channels)")
    if all(decode(units_loc[i]) == new[g] for g in new for i in np.flatnonzero(units_group == g)) and \
            all(el_loc[i] == new[el_group[i]] for i in range(len(el_group)) if el_group[i] in new):
        with h5py.File(nwb_path, "r") as h:
            if all(decode(h[f"{EPHYS}/{groups[g]}"].attrs["location"]) == new[g] for g in new):
                print("Already labelled; nothing to do.")
                return
    if args.dry_run:
        print("Dry run; file not changed.")
        return

    with h5py.File(nwb_path, "r+") as h:
        groups, units_group, _, el_group, _ = current_labels(h)
        u = h["units/location"]
        for i in np.flatnonzero(np.isin(units_group, list(new))):
            u[i] = new[int(units_group[i])]
        e = h[f"{EPHYS}/electrodes/location"]
        for i, g in enumerate(el_group):
            if g in new:
                e[i] = new[g]
        for g, label in new.items():
            a = h[f"{EPHYS}/{groups[g]}"].attrs
            a.create("location", data=label, dtype=a.get_id("location").dtype)

    # Read back through pynwb, the way nwbmatic and pynapple read it.
    with NWBHDF5IO(nwb_path, "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        bad = [(g, lab) for g, lab in new.items() if not (units.loc[units["group"] == g, "location"] == lab).all()]
        bad += [(g, lab) for g, lab in new.items() if nwb.electrode_groups[groups[g]].location != lab]
        el = nwb.electrodes.to_dataframe()
        bad += [(g, lab) for g, lab in new.items()
                if not (el.loc[[group_of_name(n) == g for n in el["group_name"]], "location"] == lab).all()]
    if bad:
        raise RuntimeError(f"Read-back through pynwb does not match for {bad}")
    print("Written and verified through pynwb.")


if __name__ == "__main__":
    main()
