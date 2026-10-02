"""
Split one epoch of a session NWB into two labelled epochs, in place.

nwbmatic and pynapple read epochs from the NWB (not from Epoch_TS.csv), so this
is enough for every downstream reader. The epochs table is stored as fixed-size
datasets, so they are deleted and rewritten with the extra row; each dataset's
NWB attributes are kept and tags_index is re-pointed at the new tags dataset.
Running it again on an already split file does nothing. A rebuilt NWB (nwbmatic
GUI) loses the split, so re-run this afterwards.

Usage:
    python SplitNWBEpoch.py <session_dir> <epoch_label> <split_time_s> <first_label> <second_label> [--dry-run]

Do not split exactly at a data boundary (e.g. the end of tracking): pynapple trims
1 us off touching epochs, so the last sample there would fall outside.

B5303-251008 (exploration and sleep recorded as one epoch; tracking ends 1504.42 s):
    python SplitNWBEpoch.py E:/B5303/B5303-251008 exploration 1506 exploration sleep
"""
import argparse
import glob
import os

import h5py
import numpy as np
from pynwb import NWBHDF5IO

EPOCHS = "intervals/epochs"


def read_epochs(h):
    g = h[EPOCHS]
    starts = g["start_time"][()]
    stops = g["stop_time"][()]
    tags = [t.decode() if isinstance(t, bytes) else t for t in g["tags"][()]]
    ends = g["tags_index"][()]
    row_tags = [tags[a:b] for a, b in zip(np.r_[0, ends[:-1]], ends)]
    return starts, stops, row_tags


def print_epochs(title, starts, stops, row_tags):
    print(title)
    for s, e, t in zip(starts, stops, row_tags):
        print(f"  {s:12.5f} {e:12.5f}  {','.join(t)}")


def replace_dataset(g, name, data, dtype):
    """Delete and recreate g[name] with new data, keeping its attributes (not 'target')."""
    old = g[name]
    saved = [(k, old.attrs[k], old.attrs.get_id(k).dtype) for k in old.attrs if k != "target"]
    del g[name]
    new = g.create_dataset(name, data=data, dtype=dtype)
    for k, v, dt in saved:
        new.attrs.create(k, data=v, dtype=dt)
    return new


def main():
    parser = argparse.ArgumentParser(description="Split one NWB epoch into two labelled epochs.")
    parser.add_argument("session_dir")
    parser.add_argument("epoch_label")
    parser.add_argument("split_time_s", type=float)
    parser.add_argument("first_label")
    parser.add_argument("second_label")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    nwb_files = glob.glob(os.path.join(args.session_dir, "pynapplenwb", "*.nwb"))
    if len(nwb_files) != 1:
        raise RuntimeError(f"Expected one NWB in {args.session_dir}/pynapplenwb, found {nwb_files}")
    nwb_path = nwb_files[0]
    t = args.split_time_s

    with h5py.File(nwb_path, "r") as h:
        starts, stops, row_tags = read_epochs(h)
    print_epochs(f"{nwb_path}\nCurrent epochs:", starts, stops, row_tags)

    # Already split?
    for i in range(1, len(starts)):
        if (np.isclose(starts[i], t) and row_tags[i] == [args.second_label]
                and row_tags[i - 1] == [args.first_label]):
            print(f"Already split at {t} s ({args.first_label} / {args.second_label}); nothing to do.")
            return

    matches = [i for i, tags in enumerate(row_tags) if tags == [args.epoch_label]]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one epoch tagged '{args.epoch_label}', found {len(matches)}")
    i = matches[0]
    if not starts[i] < t < stops[i]:
        raise RuntimeError(f"Split time {t} s is not inside {args.epoch_label} ({starts[i]}-{stops[i]} s)")

    new_starts = np.r_[starts[:i], starts[i], t, starts[i + 1:]]
    new_stops = np.r_[stops[:i], t, stops[i], stops[i + 1:]]
    new_row_tags = row_tags[:i] + [[args.first_label], [args.second_label]] + row_tags[i + 1:]
    print_epochs("New epochs:", new_starts, new_stops, new_row_tags)
    if args.dry_run:
        print("Dry run; file not changed.")
        return

    flat_tags = [tag for tags in new_row_tags for tag in tags]
    tags_index = np.cumsum([len(tags) for tags in new_row_tags])
    with h5py.File(nwb_path, "r+") as h:
        g = h[EPOCHS]
        index_dtype = g["tags_index"].dtype
        if tags_index.max() > np.iinfo(index_dtype).max:
            index_dtype = np.uint32
        replace_dataset(g, "id", np.arange(len(new_starts)), g["id"].dtype)
        replace_dataset(g, "start_time", new_starts, np.float64)
        replace_dataset(g, "stop_time", new_stops, np.float64)
        tags_ds = replace_dataset(g, "tags", flat_tags, h5py.string_dtype("utf-8"))
        index_ds = replace_dataset(g, "tags_index", tags_index, index_dtype)
        index_ds.attrs.create("target", data=tags_ds.ref, dtype=h5py.ref_dtype)

    # Read back through pynwb, the way nwbmatic and pynapple read it.
    with NWBHDF5IO(nwb_path, "r") as io:
        df = io.read().epochs.to_dataframe()
    ok = (np.allclose(df["start_time"], new_starts) and np.allclose(df["stop_time"], new_stops)
          and [list(tags) for tags in df["tags"]] == new_row_tags)
    print(df)
    if not ok:
        raise RuntimeError("Read-back through pynwb does not match the written epochs.")
    print("Written and verified through pynwb.")


if __name__ == "__main__":
    main()
