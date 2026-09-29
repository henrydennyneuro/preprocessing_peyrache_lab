"""
Per-unit running-speed, AHV and head-direction metrics over tracked wake, for
EVERY unit of a session (group assignment — TRN / ADn / other thalamus — is
done downstream from the validated unit lists). Ported from
reticular_nucleus_physiology_project (SpeedUnitMetrics.py, AHVUnitMetrics.py,
AHVTuningMetrics.py, HDUnitMetrics.py) on 260929; the computations are
unchanged except that the HD shuffles use a per-session seed (the project ran
one random stream across all sessions, which a per-session pipeline cannot
reproduce, so shuffle-based HD columns differ at Monte Carlo level).

Every table has one row per unit: session, animal, unit, location, shank,
wake_rate_hz, n_spikes, then the metrics below.

SPEED ({basename}_Speed_Tuning_Properties.csv)
  speed           kinematics.compute_speed (cm/s), clipped at SPEED_CLIP_CM
  rate            spike counts per tracking frame, Gaussian-smoothed
                  (RATE_SIGMA_S) within each contiguous tracking chunk
  speed_score     Pearson r between speed and smoothed rate over time
                  (Kropff et al. 2015-style; rate-free, sign = excited vs
                  suppressed by movement)
  null            circular shifts of the rate by >= MIN_SHIFT_S; every valid
                  shift is evaluated at once via FFT cross-correlation
  speed_sig       score outside the null's 0.5-99.5 percentiles (1%, 2-sided)
  speed_stable    score has the same sign in both halves of tracked wake
  speed_cell      speed_sig & speed_stable; speed_class = excited / suppressed
                  / n.s.
  frozen frames   ry, x and z all identical to the previous frame;
                  step_all / step_nofrozen = (moving - still rate) / mean with
                  still frames (< STILL_CM) including / excluding frozen frames
  From the speed curve CSVs: curve_mean_hz, step_index, graded_slope (rate-
  free, 4-20 cm/s, per 10 cm/s), half_curve_corr

AHV ({basename}_AHV_Tuning_Properties.csv), same test as speed:
  ahv       signed AHV (deg/s, CCW > 0), clipped at +/-AHV_CLIP_DEG
  turn_score          r(rate, |AHV|)
  turn_score_partial  r(rate, |AHV| | running speed)
  dir_score           r(rate, signed AHV): CCW (>0) vs CW (<0) preference
  *_sig / *_stable / turn_cell, turn_cell_partial, dir_cell; dir_pref
  From the AHV curve CSVs: curve_mean_hz, ahv_step ((6-90 - 0-6 deg/s) /
  mean), ahv_fast ((90-204 - 6-90) / mean), dir_slow / dir_fast ((CCW - CW) /
  mean at 6-90 / 90-204), half_curve_corr; Taube 2023/2025 slope_/r_{cw,ccw}_90,
  turn_bias, taube_shape, taube_pass (|r| >= 0.5, |slope| >= 0.05 spikes/deg,
  full + both halves; no shuffle)

HD ({basename}_HD_Information_Metrics.csv), shape-agnostic by design
(multipolar tuning must not be missed): information, split-half reliability
and movement-controlled information, not Rayleigh length.
  hd bins           N_HD_BINS over 0-2pi, tracked-wake frames
  info_raw          Skaggs information (bits/spike)
  null              N_SHIFTS circular shifts (>= MIN_SHIFT_S) of the spikes
                    against the HD trace; occupancy is shift-invariant
  info_corr         info_raw - null mean (bias-corrected); info_z; info_sig
                    (> 99th percentile)
  rayleigh          normalised first-harmonic vector length (descriptive)
  harm1..3          fraction of curve power (k = 1..3 of k = 1..10)
  half_corr         corr of circularly smoothed 1st vs 2nd half curves
  cv_ev             cross-validated explained variance (odd/even 10 s blocks,
                    100 ms counts)
  info_move_pred    HD information expected from movement alone (joint
                    speed x |AHV| lookup, averaged per HD bin)
  info_beyond_move  info_corr - info_move_pred; beyond_move_sig: exceeds the
                    shuffle margin (null p99 - null mean)
  hd_cell           info_sig & half_corr >= HALF_CORR_MIN & beyond_move_sig
  peak_deg, peak_to_mean, width_deg (FWHM of the main lobe, smoothed curve)
  info_corr_thin    corrected info after thinning to THIN_N spikes
NB this is a different raw information from the pipeline's
_HDTuning_Properties.csv spatial_information (120 bins, pynapple
compute_mutual_information), which is kept unchanged.
"""

import zlib

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress, pearsonr

# ============================
# Parameters (shared by all three metric tables)
# ============================
RATE_SIGMA_S  = 0.25
MIN_SHIFT_S   = 20.0
SIG_PCT       = (0.5, 99.5)
UNIT_CHUNK    = 16             # units per FFT batch (bounds memory in long sessions)

SPEED_CLIP_CM = 30.0
STILL_CM      = 2.0
MOVING_CM     = (4.0, 30.0)
GRADED_CM     = (2.0, 20.0)    # bin centres 3..19 cm/s

AHV_CLIP_DEG  = 204.0
STILL_DEG, SLOW_DEG = 6.0, 90.0

TAUBE_RANGES    = {"90": 90.0, "204": 204.0}
TAUBE_R_MIN     = 0.5
TAUBE_SLOPE_MIN = 0.05         # spikes/deg (Hz per deg/s), 2025 criterion
TAUBE_MIN_BINS  = 5            # minimum occupied bins to fit a side
SYM_MAX, ASYM_MIN = 0.3, 0.7

N_HD_BINS      = 60
N_SHIFTS       = 200
N_SHIFTS_THIN  = 100
THIN_N         = 1000
HALF_CORR_MIN  = 0.5
SMOOTH_BINS    = 1.0           # circular Gaussian sigma (bins) for reliability/shape metrics
HD_BLOCK_S     = 10.0
EV_BIN_S       = 0.1
MOVE_SPEED_BIN_CM = 2.0
MOVE_AHV_BIN_DEG  = 12.0
MIN_MOVE_OCC_S = 0.5
SEED           = 0


# ============================
# Shared helpers
# ============================
def unit_info(kin, basename):
    """Leading columns of every table, one dict per unit (sorted unit IDs)."""
    meta = kin.units.metadata
    units = sorted(kin.units.keys())
    rows = []
    for u in units:
        rows.append({"session": basename, "animal": basename.split("-")[0], "unit": u,
                     "location": str(meta["location"][u]) if "location" in meta else "",
                     "shank": meta["group"][u] if "group" in meta else np.nan})
    return units, rows


def frame_counts(spike_t, frame_t, fs):
    lo = np.searchsorted(spike_t, frame_t - 0.5 / fs)
    hi = np.searchsorted(spike_t, frame_t + 0.5 / fs)
    return hi - lo


def smooth_by_chunk(counts, frame_t, support, fs):
    rate = np.zeros(counts.shape, float)
    sigma = RATE_SIGMA_S * fs
    for start, end in zip(support.start, support.end):
        sel = (frame_t >= start) & (frame_t <= end)
        rate[sel] = gaussian_filter1d(counts[sel].astype(float), sigma, axis=0, mode="nearest") * fs
    return rate


def zscore(a, axis=0):
    sd = a.std(axis=axis, keepdims=True)
    return (a - a.mean(axis=axis, keepdims=True)) / np.where(sd > 0, sd, np.nan)


def circular_null(speed_z, rate_z, min_shift):
    """Pearson r between a variable and the rate circularly shifted by every valid shift."""
    n = len(speed_z)
    fs_speed = np.fft.rfft(speed_z)
    fr = np.fft.rfft(np.nan_to_num(rate_z), axis=0)
    cc = np.fft.irfft(fs_speed[:, None] * np.conj(fr), n=n, axis=0) / n    # (shifts x units)
    return cc[min_shift:n - min_shift]


def read_curves(session_dir, basename, name, occupancy_name):
    def read(suffix):
        df = pd.read_csv(session_dir / f"{basename}_{name}{suffix}.csv", index_col=0)
        df.columns = df.columns.astype(int)
        return df
    occ = pd.read_csv(session_dir / f"{basename}_{occupancy_name}.csv", index_col=0)["full"].values
    return read(""), read("_1st_half"), read("_2nd_half"), occ


# ============================
# Running speed
# ============================
def speed_curve_metrics(session_dir, basename, units):
    full, h1, h2, occ = read_curves(session_dir, basename, "Speed_Tuning_Curves", "Speed_occupancy")
    x = full.index.values
    out = {}
    for u in units:
        y = full[u].values
        ok = np.isfinite(y)
        mean = np.sum(y[ok] * occ[ok]) / occ[ok].sum() if occ[ok].sum() > 0 else np.nan
        still = y[x < STILL_CM][0]
        mv = ok & (x > MOVING_CM[0])
        moving = np.sum(y[mv] * occ[mv]) / occ[mv].sum() if occ[mv].sum() > 0 else np.nan
        g = ok & (x > GRADED_CM[0]) & (x < GRADED_CM[1])
        graded = linregress(x[g], y[g] / mean).slope * 10 if g.sum() >= 4 and mean > 0 else np.nan
        both = np.isfinite(h1[u].values) & np.isfinite(h2[u].values)
        a, b = h1[u].values[both], h2[u].values[both]
        hc = pearsonr(a, b)[0] if both.sum() > 3 and np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
        out[u] = {"curve_mean_hz": mean, "step_index": (moving - still) / mean if mean > 0 else np.nan,
                  "graded_slope": graded, "half_curve_corr": hc}
    return out


def speed_unit_metrics(kin, session_dir, basename):
    """Per-unit speed table (reads the speed curve CSVs written just before)."""
    units, info = unit_info(kin, basename)
    fs = kin.fs
    t = kin.speed.t
    speed = np.minimum(kin.speed.values, SPEED_CLIP_CM)
    frozen = kin.frozen
    spikes = kin.units
    counts = np.stack([frame_counts(spikes[u].t, t, fs) for u in units], axis=1)
    rate = smooth_by_chunk(counts, t, kin.speed.time_support, fs)

    speed_z = zscore(speed)
    rate_z = zscore(rate, axis=0)
    score = np.nanmean(speed_z[:, None] * rate_z, axis=0)
    min_shift = int(MIN_SHIFT_S * fs)
    lo, hi = np.empty(len(units)), np.empty(len(units))
    for c in range(0, len(units), UNIT_CHUNK):
        sl = slice(c, c + UNIT_CHUNK)
        lo[sl], hi[sl] = np.percentile(circular_null(speed_z, rate_z[:, sl], min_shift), SIG_PCT, axis=0)
    half = len(t) // 2
    halves = [np.nanmean(zscore(speed[s])[:, None] * zscore(rate[s], axis=0), axis=0)
              for s in (slice(0, half), slice(half, None))]

    still, moving = speed < STILL_CM, (speed > MOVING_CM[0]) & (speed <= MOVING_CM[1])
    mean_rate = counts.mean(0) * fs
    rate_still_all = counts[still].mean(0) * fs
    rate_still_nofrozen = counts[still & ~frozen].mean(0) * fs
    rate_moving = counts[moving].mean(0) * fs
    curves = speed_curve_metrics(session_dir, basename, units)

    rows = []
    for k, u in enumerate(units):
        sig = score[k] > hi[k] or score[k] < lo[k]
        stable = np.sign(halves[0][k]) == np.sign(halves[1][k]) == np.sign(score[k])
        cell = sig and stable
        rows.append({
            **info[k],
            "wake_rate_hz": mean_rate[k], "n_spikes": int(counts[:, k].sum()),
            "speed_score": score[k], "null_lo": lo[k], "null_hi": hi[k],
            "score_half1": halves[0][k], "score_half2": halves[1][k],
            "speed_sig": sig, "speed_stable": stable, "speed_cell": cell,
            "speed_class": ("excited" if score[k] > 0 else "suppressed") if cell else "n.s.",
            "frozen_frac_still": frozen[still].mean(), "frozen_frac_all": frozen.mean(),
            "step_all": (rate_moving[k] - rate_still_all[k]) / mean_rate[k] if mean_rate[k] > 0 else np.nan,
            "step_nofrozen": (rate_moving[k] - rate_still_nofrozen[k]) / mean_rate[k] if mean_rate[k] > 0 else np.nan,
            **curves[u],
        })
    return pd.DataFrame(rows)


# ============================
# Angular head velocity
# ============================
def fit_side(speed, rates, max_speed):
    keep = (speed <= max_speed) & np.isfinite(rates)
    if keep.sum() < TAUBE_MIN_BINS or np.ptp(rates[keep]) == 0:
        return np.nan, np.nan
    fit = linregress(speed[keep], rates[keep])
    return fit.slope, fit.rvalue


def side_passes(slope, r):
    return np.isfinite(slope) and abs(r) >= TAUBE_R_MIN and abs(slope) >= TAUBE_SLOPE_MIN


def fit_curve(centres, rates):
    """Slopes/r for CW and CCW over each range, and whether any side passes."""
    out, passed = {}, False
    for side, mask in (("cw", centres < 0), ("ccw", centres > 0)):
        for name, max_speed in TAUBE_RANGES.items():
            slope, r = fit_side(np.abs(centres[mask]), rates[mask], max_speed)
            out[f"slope_{side}_{name}"], out[f"r_{side}_{name}"] = slope, r
            passed |= side_passes(slope, r)
    return out, passed


def classify(s_cw, s_ccw):
    if not (np.isfinite(s_cw) and np.isfinite(s_ccw)):
        return np.nan, "unfit"
    s_max = max(abs(s_cw), abs(s_ccw))
    if s_max == 0:
        return np.nan, "flat"
    bias = abs(s_ccw - s_cw) / (2 * s_max)
    if bias < SYM_MAX:
        shape = "symmetric" if (s_cw + s_ccw) > 0 else "inverted"
    elif bias > ASYM_MIN:
        shape = "asymmetric"
    else:
        shape = "asym_unresponsive"
    return bias, shape


def partial_r(r_xy, r_xz, r_yz):
    """Correlation of x and y controlling for z."""
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))


def null_bounds(z_abs, z_dir, z_spd, rate_z, r_abs_spd, min_shift):
    """0.5/99.5 percentile bounds of the turn, partial-turn and direction nulls."""
    n_units = rate_z.shape[1]
    bounds = {k: np.empty((2, n_units)) for k in ("abs", "partial", "dir")}
    for c in range(0, n_units, UNIT_CHUNK):
        sl = slice(c, c + UNIT_CHUNK)
        null_abs = circular_null(z_abs, rate_z[:, sl], min_shift)
        null_spd = circular_null(z_spd, rate_z[:, sl], min_shift)
        bounds["abs"][:, sl] = np.nanpercentile(null_abs, SIG_PCT, axis=0)
        bounds["partial"][:, sl] = np.nanpercentile(partial_r(null_abs, r_abs_spd, null_spd), SIG_PCT, axis=0)
        del null_abs, null_spd
        bounds["dir"][:, sl] = np.nanpercentile(circular_null(z_dir, rate_z[:, sl], min_shift), SIG_PCT, axis=0)
    return bounds


def significance(score, bounds, halves):
    lo, hi = bounds
    sig = (score > hi) | (score < lo)
    stable = (np.sign(halves[0]) == np.sign(halves[1])) & (np.sign(halves[1]) == np.sign(score))
    return sig, stable, sig & stable


def half_scores(x, rate, z=None):
    half = len(x) // 2
    out = []
    for s in (slice(0, half), slice(half, None)):
        r_xr = np.nanmean(zscore(x[s])[:, None] * zscore(rate[s], axis=0), axis=0)
        if z is None:
            out.append(r_xr)
        else:
            r_zr = np.nanmean(zscore(z[s])[:, None] * zscore(rate[s], axis=0), axis=0)
            r_xz = np.corrcoef(x[s], z[s])[0, 1]
            out.append(partial_r(r_xr, r_xz, r_zr))
    return out


def ahv_curve_metrics(session_dir, basename, units):
    full, h1, h2, occ = read_curves(session_dir, basename, "AHV_Tuning_Curves", "AHV_occupancy")
    a = full.index.values
    still, slow, fast = np.abs(a) < STILL_DEG, (np.abs(a) > STILL_DEG) & (np.abs(a) < SLOW_DEG), np.abs(a) > SLOW_DEG
    out = {}
    for u in units:
        y = full[u].values
        ok = np.isfinite(y)
        mean = np.sum(y[ok] * occ[ok]) / occ[ok].sum()
        m = lambda sel: np.nanmean(y[sel]) if np.isfinite(y[sel]).any() else np.nan
        both = np.isfinite(h1[u].values) & np.isfinite(h2[u].values)
        x1, x2 = h1[u].values[both], h2[u].values[both]
        taube, pass_full = fit_curve(a, y)
        _, pass_h1 = fit_curve(a, h1[u].values)
        _, pass_h2 = fit_curve(a, h2[u].values)
        bias, shape = classify(taube["slope_cw_90"], taube["slope_ccw_90"])
        out[u] = {
            "curve_mean_hz": mean,
            "ahv_step": (m(slow) - m(still)) / mean if mean > 0 else np.nan,
            "ahv_fast": (m(fast) - m(slow)) / mean if mean > 0 else np.nan,
            "dir_slow": (m(slow & (a > 0)) - m(slow & (a < 0))) / mean if mean > 0 else np.nan,
            "dir_fast": (m(fast & (a > 0)) - m(fast & (a < 0))) / mean if mean > 0 else np.nan,
            "half_curve_corr": pearsonr(x1, x2)[0] if both.sum() > 3 and np.ptp(x1) > 0 and np.ptp(x2) > 0 else np.nan,
            **{k: taube[k] for k in ("slope_cw_90", "slope_ccw_90", "r_cw_90", "r_ccw_90")},
            "turn_bias": bias, "taube_shape": shape,
            "taube_pass": pass_full and pass_h1 and pass_h2,
        }
    return out


def ahv_unit_metrics(kin, session_dir, basename):
    """Per-unit AHV table (reads the AHV curve CSVs written just before)."""
    units, info = unit_info(kin, basename)
    fs = kin.fs
    t = kin.ahv.t
    ahv = np.clip(kin.ahv.values, -AHV_CLIP_DEG, AHV_CLIP_DEG)
    abs_ahv = np.abs(ahv)
    speed = np.minimum(kin.speed.values, SPEED_CLIP_CM)
    spikes = kin.units
    counts = np.stack([frame_counts(spikes[u].t, t, fs) for u in units], axis=1)
    rate = smooth_by_chunk(counts, t, kin.ahv.time_support, fs)
    rate_z = zscore(rate, axis=0)
    min_shift = int(MIN_SHIFT_S * fs)

    z_abs, z_dir, z_spd = zscore(abs_ahv), zscore(ahv), zscore(speed)
    r_abs = np.nanmean(z_abs[:, None] * rate_z, axis=0)
    r_dir = np.nanmean(z_dir[:, None] * rate_z, axis=0)
    r_spd = np.nanmean(z_spd[:, None] * rate_z, axis=0)
    r_abs_spd = np.corrcoef(abs_ahv, speed)[0, 1]
    r_abs_partial = partial_r(r_abs, r_abs_spd, r_spd)

    bounds = null_bounds(z_abs, z_dir, z_spd, rate_z, r_abs_spd, min_shift)
    turn = significance(r_abs, bounds["abs"], half_scores(abs_ahv, rate))
    turn_p = significance(r_abs_partial, bounds["partial"], half_scores(abs_ahv, rate, speed))
    dirn = significance(r_dir, bounds["dir"], half_scores(ahv, rate))
    curves = ahv_curve_metrics(session_dir, basename, units)
    mean_rate = counts.mean(0) * fs

    rows = []
    for k, u in enumerate(units):
        rows.append({
            **info[k],
            "wake_rate_hz": mean_rate[k], "n_spikes": int(counts[:, k].sum()),
            "absahv_speed_corr": r_abs_spd,
            "turn_score": r_abs[k], "turn_sig": turn[0][k], "turn_stable": turn[1][k], "turn_cell": turn[2][k],
            "turn_score_partial": r_abs_partial[k], "turn_partial_sig": turn_p[0][k],
            "turn_partial_stable": turn_p[1][k], "turn_cell_partial": turn_p[2][k],
            "dir_score": r_dir[k], "dir_sig": dirn[0][k], "dir_stable": dirn[1][k], "dir_cell": dirn[2][k],
            "dir_pref": ("CCW" if r_dir[k] > 0 else "CW") if dirn[2][k] else "n.s.",
            **curves[u],
        })
    return pd.DataFrame(rows)


# ============================
# Head direction
# ============================
def skaggs_counts(counts, occ_s):
    """Skaggs information (bits/spike) from spike counts per bin (bins x units)."""
    return skaggs_rates(counts / occ_s[:, None], occ_s)


def skaggs_rates(rates, occ_s):
    p = occ_s / occ_s.sum()
    mean = (p[:, None] * rates).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rates / mean
        terms = np.where(ratio > 0, p[:, None] * ratio * np.log2(ratio), 0.0)
    return terms.sum(0)


def rayleigh(rates, theta):
    return np.abs((rates * np.exp(1j * theta)[:, None]).sum(0)) / rates.sum(0)


def onehot(idx, n_cols):
    return sparse.csr_matrix((np.ones(len(idx), np.float32), (np.arange(len(idx)), idx)),
                             shape=(len(idx), n_cols))


def binned_counts(ev_frame, ev_unit, hd_bin, n_units, shift=0):
    b = hd_bin[(ev_frame + shift) % len(hd_bin)]
    return np.bincount(b * n_units + ev_unit, minlength=N_HD_BINS * n_units).reshape(N_HD_BINS, n_units)


def null_info(ev_frame, ev_unit, hd_bin, occ_s, n_units, n_shifts, rng, min_shift, theta=None):
    n = len(hd_bin)
    shifts = rng.integers(min_shift, n - min_shift, size=n_shifts)
    info = np.empty((n_shifts, n_units))
    ray = np.empty((n_shifts, n_units)) if theta is not None else None
    for s, k in enumerate(shifts):
        c = binned_counts(ev_frame, ev_unit, hd_bin, n_units, k)
        info[s] = skaggs_counts(c, occ_s)
        if theta is not None:
            ray[s] = rayleigh(c / occ_s[:, None], theta)
    return info, ray


def curve_shape(rates, theta):
    """Peak direction, peak/mean, main-lobe FWHM and harmonic power fractions of smoothed curves."""
    sm = gaussian_filter1d(rates, SMOOTH_BINS, axis=0, mode="wrap")
    peak = sm.argmax(0)
    out = {"peak_deg": np.rad2deg(theta[peak]), "peak_to_mean": sm.max(0) / sm.mean(0)}
    width = []
    for u in range(sm.shape[1]):
        c = np.roll(sm[:, u], N_HD_BINS // 2 - peak[u])       # peak to centre
        half = c.min() + (c.max() - c.min()) / 2
        above = c >= half
        centre = N_HD_BINS // 2
        lo = centre
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = centre
        while hi < N_HD_BINS - 1 and above[hi + 1]:
            hi += 1
        width.append((hi - lo + 1) * 360.0 / N_HD_BINS)
    out["width_deg"] = np.array(width)
    power = np.abs(np.fft.rfft(rates - rates.mean(0), axis=0)) ** 2
    total = power[1:11].sum(0)
    for k in (1, 2, 3):
        out[f"harm{k}"] = power[k] / total
    return out, sm


def split_half_corr(ev_frame, ev_unit, hd_bin, n_units, half_mask):
    curves = []
    for m in (half_mask, ~half_mask):
        keep = m[ev_frame]
        occ = np.bincount(hd_bin[m], minlength=N_HD_BINS).astype(float)
        c = np.bincount(hd_bin[ev_frame[keep]] * n_units + ev_unit[keep],
                        minlength=N_HD_BINS * n_units).reshape(N_HD_BINS, n_units)
        curves.append(gaussian_filter1d(c / np.maximum(occ, 1)[:, None], SMOOTH_BINS, axis=0, mode="wrap"))
    a, b = curves
    a, b = a - a.mean(0), b - b.mean(0)
    with np.errstate(invalid="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def cross_validated_ev(counts, hd_bin, t, fs):
    """Mean of odd->even and even->odd EV of 100 ms spike counts predicted by the HD curve."""
    block = ((t - t[0]) // HD_BLOCK_S).astype(int)
    tbin = ((t - t[0]) // EV_BIN_S).astype(int)
    H = onehot(hd_bin, N_HD_BINS)
    T = onehot(tbin, tbin.max() + 1)
    evs = []
    for train_odd in (True, False):
        train = (block % 2 == 1) == train_odd
        occ = np.bincount(hd_bin[train], minlength=N_HD_BINS).astype(float)
        curve = (H[train].T @ counts[train]) / np.maximum(occ, 1)[:, None]      # counts per frame, per bin
        pred = curve.astype(np.float32)[hd_bin]                                  # frames x units
        test = ~train
        obs_t = T[test].T @ counts[test]
        pred_t = T[test].T @ pred[test]
        keep = np.asarray(T[test].sum(0)).ravel() > 0
        obs_t, pred_t = obs_t[keep], pred_t[keep]
        ss_res = ((obs_t - pred_t) ** 2).sum(0)
        ss_tot = ((obs_t - obs_t.mean(0)) ** 2).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            evs.append(1 - ss_res / ss_tot)
    return np.mean(evs, axis=0)


def movement_predicted_info(counts, hd_bin, speed, abs_ahv, occ_s, fs):
    n_s = int(SPEED_CLIP_CM / MOVE_SPEED_BIN_CM)
    n_a = int(AHV_CLIP_DEG / MOVE_AHV_BIN_DEG)
    sb = np.minimum((speed / MOVE_SPEED_BIN_CM).astype(int), n_s - 1)
    ab = np.minimum((abs_ahv / MOVE_AHV_BIN_DEG).astype(int), n_a - 1)
    joint = sb * n_a + ab
    J = onehot(joint, n_s * n_a)
    occ_j = np.bincount(joint, minlength=n_s * n_a).astype(float)
    lookup = (J.T @ counts) / np.maximum(occ_j, 1)[:, None]               # counts per frame
    mean_rate = counts.mean(0)
    lookup[occ_j < MIN_MOVE_OCC_S * fs] = mean_rate
    occ_bj = (onehot(hd_bin, N_HD_BINS).T @ J).toarray()                   # hd bin x movement cell (frames)
    pred_curve = (occ_bj @ lookup) / np.maximum(occ_bj.sum(1), 1)[:, None] * fs
    return skaggs_rates(pred_curve, occ_s)


def session_rng(basename):
    """Per-session random stream, reproducible and independent of session order."""
    return np.random.default_rng([SEED, zlib.crc32(basename.encode())])


def hd_unit_metrics(kin, basename, rng=None):
    """Per-unit HD information table."""
    rng = session_rng(basename) if rng is None else rng
    units, info = unit_info(kin, basename)
    fs = kin.fs
    t = kin.ahv.t
    hd = kin.hd
    hd_bin = np.minimum((np.mod(hd, 2 * np.pi) / (2 * np.pi) * N_HD_BINS).astype(int), N_HD_BINS - 1)
    theta = (np.arange(N_HD_BINS) + 0.5) * 2 * np.pi / N_HD_BINS
    n, n_units = len(t), len(units)
    occ_s = np.bincount(hd_bin, minlength=N_HD_BINS) / fs

    ev_frame, ev_unit = [], []
    counts = np.zeros((n, n_units), np.float32)
    for k, u in enumerate(units):
        spk = kin.units[u].t
        idx = np.searchsorted(t - 0.5 / fs, spk, side="right") - 1
        ok = (idx >= 0) & (spk < t[np.clip(idx, 0, n - 1)] + 0.5 / fs)
        idx = idx[ok]
        ev_frame.append(idx); ev_unit.append(np.full(len(idx), k))
        np.add.at(counts[:, k], idx, 1)
    ev_frame, ev_unit = np.concatenate(ev_frame), np.concatenate(ev_unit)
    n_spikes = np.bincount(ev_unit, minlength=n_units)

    c_real = binned_counts(ev_frame, ev_unit, hd_bin, n_units)
    rates = c_real / occ_s[:, None]
    info_raw = skaggs_counts(c_real, occ_s)
    min_shift = int(MIN_SHIFT_S * fs)
    null, ray_null = null_info(ev_frame, ev_unit, hd_bin, occ_s, n_units, N_SHIFTS, rng, min_shift, theta)
    null_mean, null_sd, null_p99 = null.mean(0), null.std(0), np.percentile(null, 99, axis=0)
    info_corr = info_raw - null_mean
    shape, _ = curve_shape(rates, theta)
    half_corr = split_half_corr(ev_frame, ev_unit, hd_bin, n_units, np.arange(n) < n // 2)
    cv_ev = cross_validated_ev(counts, hd_bin, t, fs)
    abs_ahv = np.minimum(np.abs(kin.ahv.values), AHV_CLIP_DEG)
    speed = np.minimum(kin.speed.values, SPEED_CLIP_CM)
    info_move = movement_predicted_info(counts, hd_bin, speed, abs_ahv, occ_s, fs)
    beyond = info_corr - info_move
    beyond_sig = beyond > (null_p99 - null_mean)

    info_thin = np.full(n_units, np.nan)
    for k in range(n_units):
        if n_spikes[k] < THIN_N:
            continue
        sel = rng.choice(np.flatnonzero(ev_unit == k), THIN_N, replace=False)
        f, zeros = ev_frame[sel], np.zeros(THIN_N, int)
        c = binned_counts(f, zeros, hd_bin, 1)
        nt, _ = null_info(f, zeros, hd_bin, occ_s, 1, N_SHIFTS_THIN, rng, min_shift)
        info_thin[k] = skaggs_counts(c, occ_s)[0] - nt.mean()

    rows = []
    dur = n / fs
    for k, u in enumerate(units):
        sig = info_raw[k] > null_p99[k]
        rows.append({
            **info[k],
            "wake_rate_hz": n_spikes[k] / dur, "n_spikes": int(n_spikes[k]),
            "info_raw": info_raw[k], "info_null_mean": null_mean[k], "info_null_p99": null_p99[k],
            "info_corr": info_corr[k], "info_z": (info_raw[k] - null_mean[k]) / null_sd[k] if null_sd[k] > 0 else np.nan,
            "info_sig": sig,
            "rayleigh": rayleigh(rates, theta)[k], "rayleigh_null_p99": np.percentile(ray_null[:, k], 99),
            "half_corr": half_corr[k], "cv_ev": cv_ev[k],
            "info_move_pred": info_move[k], "info_beyond_move": beyond[k], "beyond_move_sig": beyond_sig[k],
            "hd_cell": bool(sig and half_corr[k] >= HALF_CORR_MIN and beyond_sig[k]),
            **{key: val[k] for key, val in shape.items()},
            "info_corr_thin": info_thin[k],
        })
    return pd.DataFrame(rows)
