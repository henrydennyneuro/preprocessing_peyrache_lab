"""
Check that each session's waveform files belong to the units they are indexed
by (added 260930; pipeline step check_waveform_files, batch CLI
CheckWaveformFiles.py).

_mean_wf.csv, _max_ch.csv and _waveform_parameters.csv are written by
extract_waveform_parameters from the .dat (gone for most sessions afterwards),
so they usually cannot be regenerated. Every row is checked against
{basename}_unit_ids.csv (extract_unit_ids):

  structure   same units as the NWB; 30 samples per unit at 20 kHz; channel
              count = the unit's anatomical group
  derivation  max_ch (channel with the minimum at t = 0) and trough_to_peak
              recomputed from _mean_wf.csv equal the saved files
  content     each unit's mean waveform is compared with the mean .spk snippet
              (N_SPK spikes) of every cluster on its shank, its own found via the
              uid, on the channels the anatomical and spike groups share. .spk is
              filtered and _mean_wf raw (with spike-locked LFP), so both are
              detrended per channel (quadratic over the window) and compared at
              the best alignment within +/- MAX_LAG samples of nominal:
                shape_r   Pearson r (scale-free)
                amp_err   ||wf - g * snippet|| / ||wf||, ONE gain g per session
                          (median own-cluster fit), so same-shape units of
                          different size are told apart

A session PASSES when structure and derivation are exact and, on every shank
with >= 2 units, the true mapping (row u <-> cluster of unit u) has a lower mean
amp_err than every cyclic shift of it -- the signature of a shifted file or one
extracted from a different curation. Per unit (reported, not part of the verdict):
  verified   own cluster is the lowest amp_err in its row and column, shape_r >= R_OK
  twin       shape_r >= R_OK but another cluster fits as well (near-identical units)
  weak       shape_r < R_OK: artefact-like or low-signal; listed for review
The per-channel peak-to-peak footprint r and the Hungarian best pairing are
reported alongside.

API:
  check_session(session_dir, cache=None) -> (unit table, issues, notes, shift_ok)
  summarize(basename, t, issues, notes, shift_ok) -> one-row summary (verdict ...)
  save_session(session_dir, t, verdict, ids)
      {basename}_waveform_check.csv (one row per unit, keyed by uid +
      spike_hash; own_shape_r saved as raw_vs_spk_r), stamped in
      {basename}_provenance.json; for PASS also stamps the three waveform files
  tag_parameter_files(session_dir, ids)
      appends uid + spike_hash to _max_ch.csv and _waveform_parameters.csv
      (rows = units by index); idempotent, refuses on any mismatch
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

FS = 20000.0
WF_PRE, WF_N = 10, 30          # nwbmatic waveform window: -0.5 .. +1.0 ms
SPK_PEAK = 16                  # xml peakSampleIndex (checked below)
MAX_LAG = 3                    # samples either side of the nominal alignment
N_SPK = 200                    # snippets per unit, evenly spaced over its spikes
R_OK = 0.9


def xml_groups(session_dir, basename):
    root = ET.parse(os.path.join(session_dir, f"{basename}.xml")).getroot()
    anat = [[int(c.text) for c in g.findall("channel")]
            for g in root.find("anatomicalDescription/channelGroups").findall("group")]
    spk = [([int(c.text) for c in g.find("channels").findall("channel")],
            int(g.find("nSamples").text), int(g.find("peakSampleIndex").text))
           for g in root.find("spikeDetection/channelGroups").findall("group")]
    return anat, spk


def detrended(x):
    """Remove a quadratic trend from each channel (slow LFP in the raw average)."""
    v = np.vander(np.arange(x.shape[0]), 3)
    return x - v @ np.linalg.lstsq(v, x, rcond=None)[0]


def aligned_pairs(wf, snip):
    """(lag, wf segment, snippet segment), both detrended, for each lag around nominal."""
    nominal = SPK_PEAK - WF_PRE
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        off = nominal + lag
        i0, i1 = max(0, -off), min(WF_N, snip.shape[0] - off)
        yield lag, detrended(wf[i0:i1]).ravel(), detrended(snip[i0 + off:i1 + off]).ravel()


def shape_corr(wf, snip):
    """Best Pearson r between a mean waveform (30 x ch) and a mean snippet (ns x ch) over lags."""
    return max((np.corrcoef(a, b)[0, 1], lag) for lag, a, b in aligned_pairs(wf, snip))


def own_gain(wf, snip):
    """Least-squares scale from snippet to waveform at the best-correlated lag."""
    lag = shape_corr(wf, snip)[1]
    a, b = next((a, b) for l, a, b in aligned_pairs(wf, snip) if l == lag)
    return a @ b / (b @ b)


def amp_err(wf, snip, g):
    return min(np.linalg.norm(a - g * b) / np.linalg.norm(a) for _, a, b in aligned_pairs(wf, snip))


def footprint_corr(wf, snip):
    pa, pb = np.ptp(wf, axis=0), np.ptp(snip, axis=0)
    return np.corrcoef(pa, pb)[0, 1] if len(pa) > 2 else np.nan


def load_structure(session_dir, b):
    """Structure + derivation checks; returns (ids, full waveforms, rows, issues)."""
    p = lambda suffix: os.path.join(session_dir, f"{b}{suffix}")
    ids = pd.read_csv(p("_unit_ids.csv"))
    missing = [s for s in ("_mean_wf.csv", "_max_ch.csv", "_waveform_parameters.csv") if not os.path.exists(p(s))]
    if missing:
        return ids, None, None, [f"missing {missing}"]
    wf_all = pd.read_csv(p("_mean_wf.csv"), index_col=[0, 1])
    wf_all.columns = wf_all.columns.astype(int)
    max_ch = pd.read_csv(p("_max_ch.csv"), index_col=0)["max_channel"]
    t2p = pd.read_csv(p("_waveform_parameters.csv"), index_col=0)["trough_to_peak"]
    anat, _ = xml_groups(session_dir, b)
    units = ids["unit"].to_numpy()
    issues = [f"{name} units {len(idx)} != NWB units {len(units)}"
              for name, idx in (("mean_wf", wf_all.index.get_level_values(0).unique()),
                                ("max_ch", max_ch.index), ("waveform_parameters", t2p.index))
              if not np.array_equal(np.sort(np.asarray(idx)), units)]
    if issues:
        return ids, None, None, issues
    wf, rows = {}, {}
    for r in ids.itertuples():
        w = wf_all.loc[r.unit]
        times = w.index.to_numpy()
        w = w.dropna(axis=1, how="all").to_numpy(float)
        if r.n_spikes < 1000:          # nwbmatic divides the sum by 1000 even when it has fewer spikes
            w = w * 1000 / r.n_spikes
        wf[r.unit] = w
        mc = int(np.argmin(w[WF_PRE]))                       # nwbmatic: min at t = 0
        tr = int(np.argmin(w[:, mc])); pk = tr + int(np.argmax(w[tr:, mc]))
        rows[r.unit] = {"session": b, "unit": r.unit, "uid": r.uid, "shank": r.shank, "n_spikes": r.n_spikes,
                        "location": r.location,
                        "times_ok": len(times) == WF_N and np.allclose(times, (np.arange(WF_N) - WF_PRE) / FS),
                        "n_channels_ok": w.shape[1] == len(anat[r.shank]),
                        "max_ch_ok": mc == max_ch[r.unit],
                        "t2p_ok": bool(np.isclose((pk - tr) / FS, t2p[r.unit], atol=1e-9))}
    return ids, wf, rows, []


def load_snippets(session_dir, b, ids, wf, cache):
    """Mean .spk snippet per unit and its waveform restricted to shared channels (cached)."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["wfc"].item(), z["snips"].item(), list(z["issues"]), list(z["notes"])
    p = lambda suffix: os.path.join(session_dir, f"{b}{suffix}")
    anat, spk_groups = xml_groups(session_dir, b)
    wfc, snips, issues, notes = {}, {}, [], []
    for shank, grp in ids.groupby("shank"):
        chs, ns, peak = spk_groups[shank]
        if peak != SPK_PEAK:
            issues.append(f"clu.{shank + 1}: peakSampleIndex {peak} != {SPK_PEAK}")
            continue
        labels = pd.read_csv(p(f".clu.{shank + 1}"), header=None, skiprows=1, dtype=np.int64,
                             engine="c").to_numpy().ravel()
        spk_path = p(f".spk.{shank + 1}")
        if not os.path.exists(spk_path) or os.path.getsize(spk_path) != len(labels) * ns * len(chs) * 2:
            issues.append(f"spk.{shank + 1} missing or wrong size")
            continue
        spk = np.memmap(spk_path, dtype=np.int16, mode="r", shape=(len(labels), ns, len(chs)))
        # mean_wf columns = the anatomical group's channels in xml order; the spike group can
        # leave some out (e.g. a dead channel), so compare on the channels both contain
        common = [c for c in anat[shank] if c in chs]
        if len(common) < 3:
            issues.append(f"clu.{shank + 1}: only {len(common)} channels shared by the anatomical and spike groups")
            continue
        if len(common) < len(anat[shank]):
            notes.append(f"clu.{shank + 1}: compared on {len(common)}/{len(anat[shank])} channels")
        wf_cols, spk_cols = [anat[shank].index(c) for c in common], [chs.index(c) for c in common]
        for r in grp.itertuples():
            where = np.flatnonzero(labels == r.cluster)
            pick = where[np.unique(np.linspace(0, len(where) - 1, min(N_SPK, len(where))).astype(int))]
            snips[r.unit] = spk[pick][:, :, spk_cols].astype(float).mean(axis=0)
            wfc[r.unit] = wf[r.unit][:, wf_cols]
    if cache:
        np.savez(cache, wfc=np.array(wfc, dtype=object), snips=np.array(snips, dtype=object),
                 issues=np.array(issues, dtype=object), notes=np.array(notes, dtype=object))
    return wfc, snips, issues, notes


def check_session(session_dir, cache=None):
    session_dir = str(session_dir)
    b = os.path.basename(os.path.normpath(session_dir))
    ids, wf, rows, issues = load_structure(session_dir, b)
    if issues:
        return pd.DataFrame(), issues, [], {}
    wfc, snips, issues, notes = load_snippets(session_dir, b, ids, wf, cache)
    shift_ok = {}
    if snips:
        g = np.median([own_gain(wfc[u], snips[u]) for u in snips])
        for shank, grp in ids.groupby("shank"):
            us = [u for u in grp["unit"] if u in snips]
            if not us:
                continue
            n = len(us)
            S = np.array([[shape_corr(wfc[u], snips[v])[0] for v in us] for u in us])
            E = np.array([[amp_err(wfc[u], snips[v], g) for v in us] for u in us])
            if n >= 2:
                true_cost = np.mean(np.diag(E))
                shift_costs = [np.mean(E[np.arange(n), (np.arange(n) + k) % n]) for k in range(1, n)]
                shift_ok[shank] = (bool(true_cost < min(shift_costs)), true_cost, min(shift_costs))
            r_, c_ = linear_sum_assignment(E)
            hungarian_identity = bool(np.array_equal(c_[np.argsort(r_)], np.arange(n)))
            for i, u in enumerate(us):
                e_row, e_col, s_row = np.delete(E[i], i), np.delete(E[:, i], i), np.delete(S[i], i)
                best = bool((n == 1) or (E[i, i] < e_row.min() and E[i, i] < e_col.min()))
                status = "weak" if S[i, i] < R_OK else ("verified" if best else "twin")
                rows[u].update(n_on_shank=n, gain=g, own_shape_r=S[i, i],
                               best_other_shape_r=s_row.max() if n > 1 else np.nan,
                               own_amp_err=E[i, i], best_other_amp_err=e_row.min() if n > 1 else np.nan,
                               lag=shape_corr(wfc[u], snips[u])[1],
                               own_footprint_r=footprint_corr(wfc[u], snips[u]),
                               status=status, hungarian_identity=hungarian_identity,
                               shank_shift_test=shift_ok.get(shank, (None,))[0])
    return pd.DataFrame(rows.values()), issues, notes, shift_ok



# ---------------------------------------------------------------------------
# Verdict, saving, tagging
# ---------------------------------------------------------------------------

SAVED_COLUMNS = ["session", "unit", "uid", "spike_hash", "shank", "location", "n_spikes",
                 "raw_vs_spk_r", "status", "best_other_shape_r", "own_amp_err", "best_other_amp_err",
                 "lag", "own_footprint_r", "shank_shift_test", "hungarian_identity", "gain",
                 "times_ok", "n_channels_ok", "max_ch_ok", "t2p_ok", "session_verdict"]
EXACT = ("times_ok", "n_channels_ok", "max_ch_ok", "t2p_ok")
WAVEFORM_FILES = ("_mean_wf.csv", "_max_ch.csv", "_waveform_parameters.csv")


def summarize(basename, t, issues, notes, shift_ok):
    """One-row session summary; verdict PASS needs exact structure/derivation and every shift test."""
    exact = {c: int((~t[c].astype(bool)).sum()) for c in EXACT} if len(t) else {}
    shifts_failed = [f"clu.{k + 1}" for k, v in shift_ok.items() if not v[0]]
    verdict = "PASS" if (len(t) and not issues and not any(exact.values()) and not shifts_failed
                         and "status" in t) else "FAIL"
    counts = t["status"].value_counts().to_dict() if "status" in t else {}
    weak = t[t["status"] == "weak"] if "status" in t else t.iloc[0:0]
    return {"session": basename, "verdict": verdict, "n_units": len(t),
            "verified": counts.get("verified", 0), "twin": counts.get("twin", 0), "weak": counts.get("weak", 0),
            "shift_test_failed": " ".join(shifts_failed),
            "worst_shift_margin": min((v[2] / v[1] for v in shift_ok.values()), default=np.nan),
            "exact_fails": " ".join(f"{k}={v}" for k, v in exact.items() if v),
            "issues": "; ".join(issues), "notes": "; ".join(notes),
            "weak_units": " ".join(f"{r.uid.split('.', 1)[1]}(r={r.own_shape_r:.2f})" for r in weak.itertuples())}


def save_session(session_dir, t, verdict, ids):
    """{basename}_waveform_check.csv + provenance stamps (see module docstring)."""
    from preprocessing_pipeline import unit_ids
    session_dir = str(session_dir)
    b = os.path.basename(os.path.normpath(session_dir))
    t = t.rename(columns={"own_shape_r": "raw_vs_spk_r"})
    t = t.merge(ids[["unit", "uid", "spike_hash"]], on=["unit", "uid"], how="left", validate="one_to_one")
    if t["spike_hash"].isna().any() or len(t) != len(ids):
        raise RuntimeError(f"{b}: waveform check results do not match {b}_unit_ids.csv")
    t["session_verdict"] = verdict
    path = os.path.join(session_dir, f"{b}_waveform_check.csv")
    t[[c for c in SAVED_COLUMNS if c in t]].to_csv(path, index=False)
    unit_ids.record_output(session_dir, path, ids, "check_waveform_files")
    if verdict == "PASS":
        for suffix in WAVEFORM_FILES:
            unit_ids.record_output(session_dir, f"{b}{suffix}", ids,
                                   "extract_waveform_parameters; verified by check_waveform_files")
    return path


def tag_parameter_files(session_dir, ids):
    """Append uid + spike_hash to _max_ch.csv and _waveform_parameters.csv (rows = units by index)."""
    from preprocessing_pipeline import unit_ids
    session_dir = str(session_dir)
    b = os.path.basename(os.path.normpath(session_dir))
    for suffix, col in (("_max_ch.csv", "max_channel"), ("_waveform_parameters.csv", "trough_to_peak")):
        path = os.path.join(session_dir, f"{b}{suffix}")
        df = pd.read_csv(path, index_col=0)
        if not np.array_equal(np.sort(df.index.to_numpy()), np.sort(ids["unit"].to_numpy())):
            raise RuntimeError(f"{path}: units differ from {b}_unit_ids.csv")
        tagged = unit_ids.tag(df[[col]], ids)
        if "uid" in df.columns:                       # already tagged: must agree exactly
            if not (df["uid"].tolist() == tagged["uid"].tolist()
                    and df["spike_hash"].tolist() == tagged["spike_hash"].tolist()):
                raise RuntimeError(f"{path}: existing uid/spike_hash columns do not match the current units")
            continue
        tagged.to_csv(path)
