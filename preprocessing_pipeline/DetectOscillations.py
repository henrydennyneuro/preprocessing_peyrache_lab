import os
import numpy as np
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    time_support = data.time_support
    time_index = data.as_units('s').index.values
    if type(data) is nap.TsdFrame:
        tmp = np.zeros(data.shape)
        for i, c in enumerate(data.columns):
            tmp[:, i] = bandpass_filter(data[c], lowcut, highcut, fs, order)
        return nap.TsdFrame(t=time_index, d=tmp, time_support=time_support,
                            time_units='s', columns=data.columns)
    elif type(data) is nap.Tsd:
        flfp = _butter_bandpass_filter(data.values, lowcut, highcut, fs, order)
        return nap.Tsd(t=time_index, d=flfp, time_support=time_support, time_units='s')
    else:
        raise RuntimeError("Unknown format. Should be Tsd/TsdFrame")

def detect_oscillatory_events_hilbert(lfp, epoch, freq_band, thres_band, duration_band, min_inter_duration, smoothing_bins=40):
    lfp = lfp.restrict(epoch)
    frequency = lfp.rate
    signal = bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
    envelope = np.abs(hilbert(signal.values))
    window = np.ones(smoothing_bins) / smoothing_bins
    nSS = np.convolve(envelope, window, mode='same')
    nSS = (nSS - np.mean(nSS)) / np.std(nSS)
    nSS = nap.Tsd(t=signal.index.values, d=nSS, time_support=epoch)

    nSS2 = nSS.threshold(thres_band[0], method='above')
    nSS3 = nSS2.threshold(thres_band[1], method='below')

    osc_ep = nSS3.time_support
    osc_ep = osc_ep.drop_short_intervals(duration_band[0], time_units='s')
    osc_ep = osc_ep.drop_long_intervals(duration_band[1], time_units='s')
    osc_ep = osc_ep.merge_close_intervals(min_inter_duration, time_units='s')
    osc_ep = nap.IntervalSet(osc_ep.as_dataframe().reset_index(drop=True))

    osc_max = []
    osc_tsd = []
    for count, value in enumerate(osc_ep):
        tmp = nSS.restrict(osc_ep.loc[[count]])
        osc_tsd.append(tmp.index[np.argmax(tmp)])
        osc_max.append(np.max(tmp))

    osc_tsd = nap.Tsd(t=np.array(osc_tsd), d=np.array(osc_max), time_support=epoch)

    return osc_ep, osc_tsd


if __name__ == '__main__':

    # --- Hard-coded session ---
    directory        = r'D:\B3200\B3210\B3210-240909'
    test_channel     = 29
    control_channel  = 106

    oscillation_name         = 'ripple'
    oscillation_abbreviation = 'rip'
    freq_band          = (100, 300)
    thres_band         = (3.5, 15)
    duration_band      = (0.01, 0.1)
    min_inter_duration = 0.02
    smoothing_bins     = 40
    noise_thres_band   = (4, 15)
    # --------------------------

    path_string        = Path(directory)
    recording_basename = os.path.basename(directory)

    data      = ntm.load_session(directory, "neurosuite")
    frequency = 1250
    sws_ep    = data.read_neuroscope_intervals('sws')

    print(f"sws_ep: {len(sws_ep)} intervals, "
          f"{(sws_ep['end'] - sws_ep['start']).sum():.1f} s total")

    lfp         = data.load_lfp(channel=test_channel,    extension='.eeg', frequency=frequency)
    control_lfp = data.load_lfp(channel=control_channel, extension='.eeg', frequency=frequency)

    # Control channel noise rejection
    noise_ep, _ = detect_oscillatory_events_hilbert(
        control_lfp, sws_ep, freq_band, noise_thres_band, duration_band, min_inter_duration, smoothing_bins)
    denoised_ep = sws_ep.set_diff(noise_ep)
    print(f"Control channel {control_channel}: {len(noise_ep)} noise epochs removed "
          f"({(noise_ep['end'] - noise_ep['start']).sum():.1f} s).")

    # Detect ripples on denoised epochs
    oscillation_ep, oscillation_tsd = detect_oscillatory_events_hilbert(
        lfp, sws_ep, freq_band, thres_band, duration_band, min_inter_duration, smoothing_bins)

    print(f"Found {len(oscillation_ep)} {oscillation_name}s")

    durations_ms = (oscillation_ep['end'] - oscillation_ep['start']) * 1000

    # Save .evt file for Neuroscope
    start = oscillation_ep.as_units('ms')['start'].values
    peaks = oscillation_tsd.as_units('ms').index.values
    ends  = oscillation_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start, peaks, ends)).T.flatten()
    n = len(oscillation_ep)
    texttowrite = np.vstack((
        np.repeat(np.array([f'{oscillation_abbreviation} start 1']), n),
        np.repeat(np.array([f'{oscillation_abbreviation} peak 1']),  n),
        np.repeat(np.array([f'{oscillation_abbreviation} stop 1']),  n),
    )).T.flatten()

    evt_file = os.path.join(path_string, data.basename + f'.evt.py.{oscillation_abbreviation}')
    with open(evt_file, 'w') as f:
        for t, label in zip(datatowrite, texttowrite):
            f.write("{:1.6f}".format(t) + "\t" + label + "\n")

    print(f"Saved {evt_file}")

    # --- Diagnostic plots (uncomment to use) ---

    # # Snippet: raw LFP, filtered, Hilbert envelope, z-scored
    # ex_ep        = nap.IntervalSet(start=757.036, end=758.036, time_units='s')
    # lfp_snippet  = lfp.restrict(ex_ep)
    # signal       = bandpass_filter(lfp.restrict(sws_ep), freq_band[0], freq_band[1], frequency)
    # sig_snippet  = signal.restrict(ex_ep)
    # env_snippet  = np.abs(hilbert(sig_snippet.values))
    # pyna_window  = np.ones(smoothing_bins) / smoothing_bins
    # env_smoothed = np.convolve(env_snippet, pyna_window, mode='same')
    # env_zscored  = (env_smoothed - np.mean(env_smoothed)) / np.std(env_smoothed)
    # t            = sig_snippet.as_units('s').index.values
    # fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 9))
    # axes[0].plot(t, lfp_snippet.values);  axes[0].set_ylabel('Raw LFP (a.u.)')
    # axes[1].plot(t, sig_snippet.values);  axes[1].set_ylabel('Filtered (a.u.)')
    # axes[2].plot(t, env_snippet);         axes[2].set_ylabel('Hilbert env. (a.u.)')
    # axes[3].plot(t, env_zscored);         axes[3].set_ylabel('Z-score (SD)')
    # axes[3].set_xlabel('Time (s)')
    # plt.tight_layout(); plt.show()

    # Duration histogram
    plt.figure(figsize=(8, 4))
    plt.hist(durations_ms, bins=50, edgecolor='black')
    plt.xlabel('Duration (ms)'); plt.ylabel('Count')
    plt.title(f'Ripple durations  (n={len(oscillation_ep)})')
    plt.tight_layout(); plt.show()
