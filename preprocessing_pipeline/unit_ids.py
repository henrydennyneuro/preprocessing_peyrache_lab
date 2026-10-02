"""
Stable unit identifiers, traceable to the spike sorting (added 260930).

nwbmatic numbers units with one running counter over shanks (clu files in
numeric order, clusters > 1 in ascending order within each; see
nwbmatic/neurosuite.py load_neurosuite_spikes), and the NWB keeps no cluster
number. Re-curating one shank therefore renumbers every unit on the shanks
after it, and the same (session, unit) silently names a different neuron.

This module rebuilds that numbering from the clu/res files, checks every NWB
spike train against its cluster, and gives each unit:
  uid         {session}.clu{N}.c{cluster}  (clu file number as on disk, Klusters
              cluster number) -- unaffected by curation of other shanks
  n_spikes    spikes in the NWB
  spike_hash  first 16 hex digits of sha1 over the NWB spike sample indices
              (int64 at FS); changes whenever the spike train changes, including
              when Klusters renumbers or reuses a cluster number
Tables should be joined on uid AND spike_hash; a mismatch is an error.

Per-session output: {basename}_unit_ids.csv
  session, unit (nwbmatic number), uid, shank (0-based = clu - 1), clu, cluster,
  location, n_spikes, spike_hash, n_outside_epochs (res spikes of the cluster
  outside the NWB epochs, which nwbmatic drops)

Other steps use it through:
  load_verified   the unit table, after checking it describes the spike trains
                  the step is about to use (else: re-run extract_unit_ids)
  tag             append uid + spike_hash to a one-row-per-unit table
  record_output   stamp a written per-unit file in {basename}_provenance.json
                  with units_hash (one hash over every uid + spike_hash), so
                  wide files (units as columns) are covered too
  check_output    raise unless a file carries the session's current units_hash
"""

import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

FS = 20000.0      # nwbmatic's load_neurosuite_spikes default; it does not read the xml
HASH_DIGITS = 16


class UnitIDError(RuntimeError):
    """The NWB units do not match the clu/res files."""


def spike_hash(samples):
    return hashlib.sha1(np.asarray(samples, dtype="<i8").tobytes()).hexdigest()[:HASH_DIGITS]


def make_uid(session, clu, cluster):
    return f"{session}.clu{clu}.c{cluster}"


def _read_ints(path, skip_header=False):
    return pd.read_csv(path, header=None, skiprows=1 if skip_header else 0,
                       dtype=np.int64, engine="c").to_numpy().ravel()


def session_nwb(session_dir):
    """The single NWB in pynapplenwb/; more than one is an error (loaders take the first)."""
    files = sorted(glob.glob(os.path.join(session_dir, "pynapplenwb", "*.nwb")))
    if len(files) != 1:
        raise UnitIDError(f"Expected one NWB in {session_dir}/pynapplenwb, found {len(files)}: {files}")
    return files[0]


def _nwb_units(nwb_path):
    with h5py.File(nwb_path, "r") as h:
        u = h["units"]
        ids = u["id"][()]
        ends = u["spike_times_index"][()]
        times = u["spike_times"][()]
        group = u["group"][()]
        location = [x.decode() if isinstance(x, bytes) else x for x in u["location"][()]]
        epochs = np.column_stack([h["intervals/epochs/start_time"][()],
                                  h["intervals/epochs/stop_time"][()]])
    trains = np.split(times, ends[:-1])
    return ids, trains, group, location, epochs


def _outside(t, epochs, tol):
    inside = np.zeros(len(t), bool)
    for s, e in epochs:
        inside |= (t >= s - tol) & (t <= e + tol)
    return ~inside


def build_unit_ids(session_dir):
    """Unit table for one session; raises UnitIDError on any mismatch with the clu/res files."""
    session_dir = str(session_dir)
    basename = os.path.basename(os.path.normpath(session_dir))
    nwb_path = session_nwb(session_dir)

    xml_fs = float(ET.parse(os.path.join(session_dir, f"{basename}.xml"))
                   .getroot().find("acquisitionSystem/samplingRate").text)
    if xml_fs != FS:
        raise UnitIDError(f"{basename}: xml samplingRate {xml_fs} != nwbmatic's fixed {FS}")

    def suffixes(kind):
        return sorted(int(f.rsplit(".", 1)[-1]) for f in os.listdir(session_dir)
                      if f.startswith(f"{basename}.{kind}.") and f.rsplit(".", 1)[-1].isdigit())
    clus = suffixes("clu")
    if clus != suffixes("res"):
        raise UnitIDError(f"{basename}: clu files {clus} != res files {suffixes('res')}")

    ids, trains, group, location, epochs = _nwb_units(nwb_path)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise UnitIDError(f"{basename}: NWB unit ids are not 0..{len(ids) - 1}")

    rows, unit = [], 0
    for clu in clus:
        labels = _read_ints(os.path.join(session_dir, f"{basename}.clu.{clu}"), skip_header=True)
        res = _read_ints(os.path.join(session_dir, f"{basename}.res.{clu}"))
        if len(labels) != len(res):
            raise UnitIDError(f"{basename}: clu.{clu} has {len(labels)} labels, res.{clu} {len(res)} spikes")
        if labels.max() <= 1:
            continue                                   # noise / MUA only: no units (as nwbmatic)
        for cluster in np.unique(labels[labels > 1]):
            if unit >= len(ids):
                raise UnitIDError(f"{basename}: clu/res have more units than the NWB ({len(ids)})")
            where = f"{basename} unit {unit} (clu.{clu} cluster {cluster})"
            if group[unit] != clu - 1:
                raise UnitIDError(f"{where}: NWB group {group[unit]} != clu - 1 = {clu - 1}")
            t = trains[unit]
            samples = np.rint(t * FS).astype(np.int64)
            if len(t) and np.max(np.abs(t * FS - samples)) > 1e-3:
                raise UnitIDError(f"{where}: NWB spike times are not on the {FS:.0f} Hz sample grid")
            res_c = res[labels == cluster]
            if not np.isin(samples, res_c).all():
                raise UnitIDError(f"{where}: NWB spikes not in the cluster "
                                  f"({np.sum(~np.isin(samples, res_c))} of {len(samples)})")
            dropped = res_c[~np.isin(res_c, samples)]
            if not _outside(dropped / FS, epochs, tol=1.5 / FS).all():
                raise UnitIDError(f"{where}: {len(dropped)} cluster spikes missing from the NWB, "
                                  f"some inside the epochs")
            rows.append({"session": basename, "unit": unit, "uid": make_uid(basename, clu, cluster),
                         "shank": clu - 1, "clu": clu, "cluster": int(cluster), "location": location[unit],
                         "n_spikes": len(samples), "spike_hash": spike_hash(samples),
                         "n_outside_epochs": len(dropped)})
            unit += 1
    if unit != len(ids):
        raise UnitIDError(f"{basename}: clu/res give {unit} units, the NWB has {len(ids)}")
    table = pd.DataFrame(rows)
    if table["spike_hash"].duplicated().any():
        raise UnitIDError(f"{basename}: two units share a spike train (duplicate spike_hash)")
    return table


# ---------------------------------------------------------------------------
# Used by the other steps
# ---------------------------------------------------------------------------

def _paths(session_dir):
    session_dir = str(session_dir)
    basename = os.path.basename(os.path.normpath(session_dir))
    return (os.path.join(session_dir, f"{basename}_unit_ids.csv"),
            os.path.join(session_dir, f"{basename}_provenance.json"))


def units_hash(table):
    """One hash over the whole unit set (uid + spike_hash, in unit order)."""
    text = "\n".join(f"{u}:{h}" for u, h in zip(table["uid"], table["spike_hash"]))
    return hashlib.sha1(text.encode()).hexdigest()[:HASH_DIGITS]


def load_verified(session_dir, spikes):
    """The session's unit table, after checking it describes these spike trains (a TsGroup)."""
    ids_path, _ = _paths(session_dir)
    if not os.path.exists(ids_path):
        raise UnitIDError(f"{ids_path} missing: run extract_unit_ids first")
    table = pd.read_csv(ids_path)
    if not np.array_equal(np.asarray(list(spikes.keys())), table["unit"].to_numpy()):
        raise UnitIDError(f"{ids_path}: units differ from the loaded spikes; re-run extract_unit_ids")
    for u, h in zip(table["unit"], table["spike_hash"]):
        if spike_hash(np.rint(spikes[u].times() * FS).astype(np.int64)) != h:
            raise UnitIDError(f"{ids_path}: unit {u} spike train changed since extract_unit_ids; re-run it")
    return table


def tag(df, table, unit_col=None):
    """Append uid and spike_hash to a one-row-per-unit table, matched on unit_col (or the index)."""
    units = df[unit_col].to_numpy() if unit_col else df.index.to_numpy()
    by_unit = table.set_index("unit")
    out = df.copy()
    out["uid"] = by_unit.loc[units, "uid"].to_numpy()
    out["spike_hash"] = by_unit.loc[units, "spike_hash"].to_numpy()
    return out


def record_output(session_dir, filename, table, step):
    """Stamp a written per-unit file with the unit set it was computed from."""
    _, prov_path = _paths(session_dir)
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {}
    prov[os.path.basename(filename)] = {"units_hash": units_hash(table), "step": step,
                                        "written": datetime.now().isoformat(timespec="seconds")}
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True)


def check_output(session_dir, filename):
    """Raise unless filename was stamped with the session's current unit set."""
    ids_path, prov_path = _paths(session_dir)
    name = os.path.basename(filename)
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {}
    if name not in prov:
        raise UnitIDError(f"{name}: no provenance stamp in {prov_path}")
    current = units_hash(pd.read_csv(ids_path))
    if prov[name]["units_hash"] != current:
        raise UnitIDError(f"{name} was computed from a different unit set "
                          f"({prov[name]['units_hash']} vs current {current}); regenerate it")
    return prov[name]
