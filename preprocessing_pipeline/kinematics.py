"""
Kinematics over tracked wake: head direction, angular head velocity (AHV) and
running speed, computed identically for every tuning step of the pipeline.
Ported unchanged from reticular_nucleus_physiology_project/ahv_core.py (260929)
so that the pipeline reproduces the numbers of the TRN project analyses.

AHV follows the Taube lab (Graham et al. 2023 J Neurosci; LMN/DTN 2025 Nat
Commun): HD smoothed with a running average, then AHV = least-squares slope
over the same window. Their 5-point window at 60 Hz spans 4 frame intervals
(66.7 ms); the window here is matched in time, not samples (9 points at
120 Hz, 7 at 100 Hz). Sign convention: CCW turns positive, CW negative.

Tracking frame (Motive, right-handed, Y up): y is vertical, the floor is the
x-z plane. Increasing ry is a positive rotation about +Y, i.e. CCW viewed
from above (a leftward turn). Verified empirically per session by
heading_turn_corr(): while running, changes in ry correlate positively with
changes in travel heading atan2(-dz, dx) (checked 260923 on B3225-260803,
B3213-241009, B5303-251014, B2902-230601: r = +0.30 to +0.47). A negative
value would mean CW/CCW (and ipsi/contra) labels are flipped for that session.

Sessions are read with nap.load_file (read-only) rather than through the
nwbmatic session object: the tracked-wake definition below (epoch tags
containing 'wake' or 'explor', intersected with the stored
position_time_support) is the one the published numbers use.
"""

import numpy as np
import pynapple as nap
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

WAKE_KEYWORDS  = ("wake", "explor")   # matches Wake, exploration, soundexploration
SMOOTH_SPAN_S  = 4 / 60.0             # Taube: 5 points at 60 Hz = 4 intervals
MAX_GAP_FRAMES = 1.5                  # split tracking where dt > 1.5 frame intervals
HEADING_STEP_S = 0.25                 # step for travel-heading changes in heading_turn_corr
RUNNING_CM_S   = 10.0                 # running threshold for heading_turn_corr


def load_wake_hd(nwb_path):
    """
    Returns (data, ry restricted to tracked wake, wake_ep, fs), or None if the
    session has no wake epoch or no head direction.
    """
    data = nap.load_file(str(nwb_path))
    if "ry" not in data.keys():
        return None
    # nap.load_file merges tracking segments into one time_support; use the stored one
    tracking_ep = data["position_time_support"] if "position_time_support" in data.keys() \
        else data["ry"].time_support
    epochs = data["epochs"]
    is_wake = np.array([any(k in str(tag).lower() for k in WAKE_KEYWORDS) for tag in epochs.tags])
    if not is_wake.any():
        return None
    wake_ep = epochs[is_wake].intersect(tracking_ep)
    ry = data["ry"].restrict(wake_ep)
    fs = 1.0 / np.median(np.diff(ry.t))
    return data, ry, wake_ep, fs


def smoothing_window(fs):
    """Odd window whose span (n - 1 intervals) matches SMOOTH_SPAN_S."""
    return 2 * int(round(SMOOTH_SPAN_S * fs / 2)) + 1


def contiguous_chunks(t, fs):
    breaks = np.where(np.diff(t) > MAX_GAP_FRAMES / fs)[0] + 1
    return np.split(np.arange(len(t)), breaks)


def compute_ahv(ry, fs):
    """
    Returns (ahv, raw_ahv, frozen_frac):
      ahv         nap.Tsd of smoothed AHV (deg/s); time_support covers only the
                  valid (edge-trimmed) samples of each contiguous tracking chunk
      raw_ahv     np.ndarray of raw frame-difference AHV (deg/s), for comparison
      frozen_frac fraction of frame-to-frame HD steps that are exactly zero
    """
    t, hd = ry.t, ry.values
    n_win = smoothing_window(fs)
    half = n_win // 2
    t_parts, ahv_parts, raw_parts, starts, ends = [], [], [], [], []
    n_frozen = n_diffs = 0

    for idx in contiguous_chunks(t, fs):
        chunk_t, chunk_hd = t[idx], hd[idx]
        good = ~np.isnan(chunk_hd)
        chunk_t, chunk_hd = chunk_t[good], chunk_hd[good]
        if len(chunk_hd) <= n_win:
            continue
        dt = np.median(np.diff(chunk_t))
        unwrapped = np.unwrap(chunk_hd)

        steps = np.diff(unwrapped)
        n_frozen += np.sum(steps == 0)
        n_diffs += len(steps)
        raw_parts.append(np.rad2deg(steps / dt))

        smoothed = uniform_filter1d(unwrapped, size=n_win, mode="nearest")
        slope = savgol_filter(smoothed, window_length=n_win, polyorder=1, deriv=1, delta=dt)
        t_valid = chunk_t[half:-half]            # drop edge-padded samples
        t_parts.append(t_valid)
        ahv_parts.append(np.rad2deg(slope[half:-half]))
        starts.append(t_valid[0] - 0.5 / fs)
        ends.append(t_valid[-1] + 0.5 / fs)

    if not t_parts:
        return None, np.array([]), np.nan
    support = nap.IntervalSet(start=starts, end=ends)
    ahv = nap.Tsd(t=np.concatenate(t_parts), d=np.concatenate(ahv_parts), time_support=support)
    frozen = n_frozen / n_diffs if n_diffs else np.nan
    return ahv, np.concatenate(raw_parts), frozen


def compute_speed(data, ry, fs):
    """
    Floor-plane (x-z) running speed in cm/s, computed exactly like AHV
    (Taube 2023 computed linear velocity with the same 5-point smoothing and
    best-fit slope): position smoothed with the time-matched running average,
    velocity = least-squares slope over the same window, speed = |velocity|.
    Uses the same contiguous chunks and edge trimming as compute_ahv, so the
    returned Tsd has the same timestamps and time_support as the AHV Tsd.
    """
    n_win = smoothing_window(fs)
    half = n_win // 2
    t_parts, speed_parts, starts, ends = [], [], [], []
    for idx in contiguous_chunks(ry.t, fs):
        chunk_t = ry.t[idx]
        good = ~np.isnan(ry.values[idx])
        chunk_t = chunk_t[good]
        if len(chunk_t) <= n_win:
            continue
        dt = np.median(np.diff(chunk_t))
        ep = nap.IntervalSet(start=chunk_t[0], end=chunk_t[-1])
        x = data["x"].restrict(ep)
        z = data["z"].restrict(ep)
        keep = np.isin(x.t, chunk_t)
        xv, zv = x.values[keep], z.values[keep]
        vel = []
        for v in (xv, zv):
            smoothed = uniform_filter1d(v, size=n_win, mode="nearest")
            vel.append(savgol_filter(smoothed, window_length=n_win, polyorder=1, deriv=1, delta=dt))
        t_valid = chunk_t[half:-half]
        t_parts.append(t_valid)
        speed_parts.append(np.hypot(*vel)[half:-half] * 100)
        starts.append(t_valid[0] - 0.5 / fs)
        ends.append(t_valid[-1] + 0.5 / fs)
    if not t_parts:
        return None
    support = nap.IntervalSet(start=starts, end=ends)
    return nap.Tsd(t=np.concatenate(t_parts), d=np.concatenate(speed_parts), time_support=support)


def frozen_mask(data, frame_t):
    """True where ry, x and z are all identical to the preceding raw frame."""
    ry = data["ry"]
    idx = np.searchsorted(ry.t, frame_t)
    idx = np.clip(idx, 1, len(ry.t) - 1)
    same = np.ones(len(frame_t), bool)
    for key in ("ry", "x", "z"):
        v = data[key].values
        same &= v[idx] == v[idx - 1]
    return same


def heading_turn_corr(data, ry, fs):
    """
    Correlation, while running, between changes in ry and changes in travel
    heading atan2(-dz, dx) (the positive-about-+Y, CCW-from-above frame).
    Positive = ry increases CCW viewed from above, as assumed throughout.
    """
    n = max(1, int(round(HEADING_STEP_S * fs)))
    d_hd, d_heading = [], []
    for idx in contiguous_chunks(ry.t, fs):
        if len(idx) < 3 * n:
            continue
        t = ry.t[idx]
        ep = nap.IntervalSet(start=t[0], end=t[-1])
        x, z = data["x"].restrict(ep).values, data["z"].restrict(ep).values
        hd = np.unwrap(ry.values[idx])
        if len(x) != len(hd):
            continue
        x, z, hd = uniform_filter1d(x, n), uniform_filter1d(z, n), uniform_filter1d(hd, n)
        dx, dz = x[n:] - x[:-n], z[n:] - z[:-n]
        speed = np.hypot(dx, dz) / (n / fs) * 100
        heading = np.unwrap(np.arctan2(-dz, dx))
        dh = heading[n:] - heading[:-n]
        dry = (hd[n:] - hd[:-n])[n:]
        running = speed[n:] > RUNNING_CM_S
        ok = running & (np.abs(dh) < np.pi / 2)
        d_hd.append(dry[ok]); d_heading.append(dh[ok])
    if not d_hd:
        return np.nan
    a, b = np.concatenate(d_hd), np.concatenate(d_heading)
    return np.corrcoef(a, b)[0, 1] if len(a) > 10 else np.nan


class TrackedWake:
    """
    Everything the tuning steps share for one session, computed once:
      data, ry (tracked wake), wake_ep, fs     from load_wake_hd
      ahv, speed                               nap.Tsd (deg/s, cm/s) on the same
                                               edge-trimmed frames / time_support
      frozen                                   bool per frame (frozen_mask)
      hd                                       ry value at each frame (rad)
      frozen_frac                              fraction of zero HD steps
    Build with TrackedWake.load(nwb_path); returns None when the session has
    no head direction, no wake epoch or no usable tracking in wake.
    """

    def __init__(self, data, ry, wake_ep, fs, ahv, speed, frozen_frac):
        self.data, self.ry, self.wake_ep, self.fs = data, ry, wake_ep, fs
        self.ahv, self.speed, self.frozen_frac = ahv, speed, frozen_frac
        assert np.array_equal(ahv.t, speed.t), "AHV and speed timestamps differ"
        self.hd = ry.values[np.searchsorted(ry.t, ahv.t)]
        self.frozen = frozen_mask(data, ahv.t)

    @classmethod
    def load(cls, nwb_path):
        loaded = load_wake_hd(nwb_path)
        if loaded is None:
            return None
        data, ry, wake_ep, fs = loaded
        ahv, _, frozen_frac = compute_ahv(ry, fs)
        speed = compute_speed(data, ry, fs)
        if ahv is None or speed is None:
            return None
        return cls(data, ry, wake_ep, fs, ahv, speed, frozen_frac)

    @property
    def units(self):
        return self.data["units"]

    def as_tsdframe(self):
        """hd (rad), ahv (deg/s, CCW > 0), speed (cm/s), frozen (0/1) on the tracked-wake frames."""
        return nap.TsdFrame(t=self.ahv.t,
                            d=np.column_stack([self.hd, self.ahv.values, self.speed.values,
                                               self.frozen.astype(float)]),
                            columns=["hd", "ahv", "speed", "frozen"],
                            time_support=self.ahv.time_support)
