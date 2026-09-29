"""
1D self-motion tuning curves (running speed, AHV), computed, split, saved and
plotted identically. Ported unchanged from reticular_nucleus_physiology_project
(tuning_core.py; the curve parameters and per-region figures of
SpeedTuningCurves.py / AHVTuningCurves.py) on 260929.

  - tuning curves with an occupancy criterion (bins < MIN_OCCUPANCY_S -> NaN)
  - full session, 1st/2nd half of tracked wake time, alternating 10 s blocks
  - one grid figure per region (ADn, TRN), one panel per unit

Per-session outputs (rows = bin centres, columns = unit IDs):
  {basename}_Speed_Tuning_Curves{,_1st_half,_2nd_half,_odd_bins,_even_bins}.csv
      index 'speed_cm_s', 2 cm/s bins over 0-30 cm/s
  {basename}_AHV_Tuning_Curves{...}.csv
      index 'ahv_deg_s', 6 deg/s bins over -204..+204 deg/s (CW < 0 < CCW)
  {basename}_Speed_occupancy.csv / _AHV_occupancy.csv  seconds per bin per split
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pynapple as nap

matplotlib.use("Agg")

MIN_OCCUPANCY_S = 0.5     # Taube: 30 samples at 60 Hz
BLOCK_S         = 10.0    # alternating-block duration for odd/even curves
REGIONS = {"ADn": "ADN", "TRN": "TRN"}   # figure label -> upper-cased 'location' label
SPLIT_SUFFIXES = {"": "full", "_1st_half": "1st_half", "_2nd_half": "2nd_half",
                  "_odd_bins": "odd_bins", "_even_bins": "even_bins"}

SPEED_BIN_WIDTH_CM = 2.0
SPEED_MAX_CM       = 30.0
AHV_BIN_WIDTH_DEG  = 6.0
AHV_MAX_DEG        = 204.0
AHV_SLOW_RANGE_DEG = 90.0    # Taube 0-90 deg/s range, marked on plots

SPEED_EDGES = np.arange(0, SPEED_MAX_CM + SPEED_BIN_WIDTH_CM / 2, SPEED_BIN_WIDTH_CM)
AHV_EDGES   = np.arange(-AHV_MAX_DEG, AHV_MAX_DEG + AHV_BIN_WIDTH_DEG / 2, AHV_BIN_WIDTH_DEG)

COLOUR_REF   = "#8a8984"
TEXT_MUTED   = "#52514e"
COLOUR_SPEED = "#1baf7a"
COLOUR_CW    = "#2a78d6"
COLOUR_CCW   = "#eb6834"


def split_halves(ep):
    """Split an IntervalSet into two halves of equal total duration."""
    durations = ep.end - ep.start
    cum = np.cumsum(durations)
    half = cum[-1] / 2
    i = np.searchsorted(cum, half)
    t_split = ep.end[i] - (cum[i] - half)
    first = ep.intersect(nap.IntervalSet(start=ep.start[0], end=t_split))
    second = ep.intersect(nap.IntervalSet(start=t_split, end=ep.end[-1]))
    return first, second


def split_blocks(ep):
    """Alternating BLOCK_S blocks (odd = 1st, 3rd, ...; even = 2nd, 4th, ...)."""
    edges = np.arange(ep.start[0], ep.end[-1] + BLOCK_S, BLOCK_S)
    blocks = nap.IntervalSet(start=edges[:-1], end=edges[1:])
    odd = blocks[np.arange(0, len(blocks), 2)].intersect(ep)
    even = blocks[np.arange(1, len(blocks), 2)].intersect(ep)
    return odd, even


def tuning_curves(spikes, feature, ep, edges, fs, index_name):
    """Returns (DataFrame bins x units in Hz, occupancy seconds per bin)."""
    feature = feature.restrict(ep)
    tc = nap.compute_tuning_curves(spikes, feature, bins=[edges], epochs=ep, fs=fs)
    occ_s = np.histogram(feature.values, bins=edges)[0] / fs
    centres = (edges[:-1] + edges[1:]) / 2
    df = pd.DataFrame(np.asarray(tc).T, index=pd.Index(centres, name=index_name),
                      columns=list(spikes.keys()))
    df[occ_s < MIN_OCCUPANCY_S] = np.nan
    return df, occ_s


def compute_and_save_splits(spikes, feature, edges, fs, index_name, session_dir, basename,
                            curves_name, occupancy_name):
    """
    Writes {basename}_{curves_name}{suffix}.csv for every split in
    SPLIT_SUFFIXES and {basename}_{occupancy_name}.csv (seconds per bin, one
    column per split). Returns (full-session curves, full-session occupancy).
    """
    ep = feature.time_support
    first, second = split_halves(ep)
    odd, even = split_blocks(ep)
    split_eps = {"": ep, "_1st_half": first, "_2nd_half": second, "_odd_bins": odd, "_even_bins": even}

    occupancy, full_tc = {}, None
    for suffix, split_ep in split_eps.items():
        tc, occ_s = tuning_curves(spikes, feature, split_ep, edges, fs, index_name)
        tc.to_csv(session_dir / f"{basename}_{curves_name}{suffix}.csv")
        occupancy[SPLIT_SUFFIXES[suffix]] = occ_s
        if suffix == "":
            full_tc = tc
    pd.DataFrame(occupancy, index=full_tc.index).to_csv(session_dir / f"{basename}_{occupancy_name}.csv")
    return full_tc, occupancy["full"]


def unit_locations(spikes):
    if "location" not in spikes.metadata:
        return {}
    return spikes.metadata["location"].astype(str).to_dict()


def wake_rates_from_curves(tc, occ_s):
    """Mean rate over occupied bins, weighted by occupancy (Hz)."""
    valid = np.isfinite(tc.values)
    weights = np.where(valid, occ_s[:, None], 0.0)
    return pd.Series(np.nansum(tc.values * weights, axis=0) / weights.sum(axis=0), index=tc.columns)


# ============================
# Per-region grid figures
# ============================
def plot_by_region(tc, locations, occ_s, plots_dir, file_stem, title_fn, draw_unit, xlabel, handles):
    """
    One figure per region: {file_stem}_{region}.png. title_fn(region, n_units,
    wake_s) returns the two-line title; draw_unit(ax, rates) draws one panel.
    """
    wake_rates = wake_rates_from_curves(tc, occ_s)
    for region, match in REGIONS.items():
        units = sorted(u for u in tc.columns if str(locations.get(u, "")).strip().upper() == match)
        if not units:
            continue
        plot_unit_grid(tc, units, draw_unit, wake_rates, title_fn(region, len(units), occ_s.sum()),
                       xlabel, handles, plots_dir / f"{file_stem}_{region}.png")


def plot_unit_grid(tc, units, draw_unit, wake_rates, title, xlabel, handles, save_path):
    n = len(units)
    n_cols = max(4, int(np.ceil(np.sqrt(n * 1.4))))   # min width keeps title + legend clear
    n_rows = int(np.ceil(n / n_cols))
    title_in = 0.6   # figure height reserved for the two-line title
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 1.9 + title_in),
                             squeeze=False)

    for ax, unit in zip(axes.flat, units):
        rates = tc[unit].values
        draw_unit(ax, rates)
        top = np.nanmax(rates) if np.isfinite(rates).any() else 1.0
        ax.set_ylim(0, top * 1.15 if top > 0 else 1.0)
        ax.set_title(f"{unit} · {wake_rates[unit]:.1f} Hz", fontsize=7, color=TEXT_MUTED, pad=2)
        ax.tick_params(labelsize=6, length=2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate (Hz)", fontsize=7)

    if handles:
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, ncol=len(handles))
    fig.suptitle(title, fontsize=9, x=0.01, ha="left", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - title_in / fig.get_figheight()))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_speed_by_region(tc, locations, occ_s, basename, plots_dir):
    """One figure per region: {basename}_speed_tuning_curves_{region}.png."""
    centres = tc.index.values

    def draw(ax, rates):
        ax.plot(centres, rates, color=COLOUR_SPEED, linewidth=1.5)
        ax.set_xlim(0, SPEED_MAX_CM)

    def title(region, n_units, wake_s):
        return (f"{basename} — {region} running-speed tuning (n = {n_units})\n"
                f"{SPEED_BIN_WIDTH_CM:.0f} cm/s bins, ≥{MIN_OCCUPANCY_S} s occupancy; "
                f"{wake_s:.0f} s wake within 0–{SPEED_MAX_CM:.0f} cm/s")

    plot_by_region(tc, locations, occ_s, plots_dir, f"{basename}_speed_tuning_curves",
                   title, draw, "Speed (cm/s)", handles=None)


def plot_ahv_by_region(tc, locations, occ_s, basename, plots_dir):
    """One figure per region: {basename}_AHV_tuning_curves_{region}.png (CW and CCW vs |AHV|)."""
    centres = tc.index.values
    speed = np.abs(centres)
    cw, ccw = centres < 0, centres > 0

    def draw(ax, rates):
        ax.plot(speed[cw][::-1], rates[cw][::-1], color=COLOUR_CW, linewidth=1.5)
        ax.plot(speed[ccw], rates[ccw], color=COLOUR_CCW, linewidth=1.5)
        ax.axvline(AHV_SLOW_RANGE_DEG, color=COLOUR_REF, linewidth=0.6, linestyle=":")
        ax.set_xlim(0, AHV_MAX_DEG)

    def title(region, n_units, wake_s):
        return (f"{basename} — {region} AHV tuning (n = {n_units})\n"
                f"6°/s bins, ≥{MIN_OCCUPANCY_S} s occupancy; {wake_s:.0f} s wake within "
                f"±{AHV_MAX_DEG:.0f}°/s; dotted = {AHV_SLOW_RANGE_DEG:.0f}°/s")

    handles = [plt.Line2D([], [], color=COLOUR_CW, linewidth=1.5, label="CW turns"),
               plt.Line2D([], [], color=COLOUR_CCW, linewidth=1.5, label="CCW turns")]
    plot_by_region(tc, locations, occ_s, plots_dir, f"{basename}_AHV_tuning_curves",
                   title, draw, "|AHV| (°/s)", handles)
