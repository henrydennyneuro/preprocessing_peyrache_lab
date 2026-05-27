import os
import yaml
import json
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pathlib import Path
from scipy.signal import butter, lfilter, filtfilt, hilbert
from scipy.optimize import curve_fit

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def _bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def _bandpass_filter_tsd(tsd, lowcut, highcut, fs, order=4):
    flfp = _bandpass_filter(tsd.values, lowcut, highcut, fs, order)
    return nap.Tsd(t=tsd.as_units('s').index.values, d=flfp,
                   time_support=tsd.time_support, time_units='s')

def detect_oscillatory_events_hilbert(lfp, epoch, freq_band, thres_band, duration_band, min_inter_duration, smoothing_bins=40):
    lfp = lfp.restrict(epoch)
    signal = _bandpass_filter_tsd(lfp, freq_band[0], freq_band[1], lfp.rate)
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


class PreprocessingPipeline:
    def __init__(self):
        """Initialize the preprocessing pipeline without assuming a fixed data directory."""
        pass

    def load_yaml_config(self, config_path):
        """Load the YAML configuration file specifying files and steps."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        files = config.get('files', [])
        steps = config.get('steps', [])
        delete_dat = config.get('delete_dat', True)
        return files, steps, delete_dat

    def process_from_yaml(self, config_path):
        """Process files and steps specified in a YAML configuration."""
        files, steps, delete_dat = self.load_yaml_config(config_path)
        for file in files:
            full_path = Path(file)
            if full_path.exists():
                print(f"Processing file: {full_path}")
                self.process_recording(directory=full_path, steps=steps, delete_dat=delete_dat)
            else:
                print(f"File not found: {full_path}")

    def process_recording(self, directory, steps, delete_dat=True):
        """Main function to process a single recording with selected steps."""
        path_string = Path(directory)
        recording_basename = os.path.basename(directory)
        print(f'Processing recording: {recording_basename}')

        # Load session data
        data = ntm.load_session(directory, "neurosuite")

        # Dynamically call specified steps
        for step in steps:
            if hasattr(self, step):
                method = getattr(self, step)
                print(f"Executing step: {step}")
                method(data, path_string, recording_basename)
            else:
                raise ValueError(f"Invalid preprocessing step: {step}")

        if delete_dat:
            dat_file = path_string / f"{recording_basename}.dat"
            if dat_file.exists():
                dat_file.unlink()
                print(f"Deleted {dat_file}")
            else:
                print(f"dat file not found, skipping deletion: {dat_file}")

        print(f'Finished processing: {recording_basename}')

    # Preprocessing Methods
    def extract_inter_spike_intervals(self, data, path_string, recording_basename):
        """Extract inter-spike intervals for all neurons."""
        spikes = data.spikes
        sleep_ep = data.epochs['Sleep']
        sleep_spikes = spikes.restrict(sleep_ep)

        median_isis = []
        median_sleep_isis = []

        for neuron in spikes:
            median_isis.append(self.calculate_mean_isi(spikes[neuron].times()))
            median_sleep_isis.append(self.calculate_mean_isi(sleep_spikes[neuron].times()))

        pd.DataFrame({'median_isi': median_isis, 'median_sleep_isi': median_sleep_isis}).to_csv(
            os.path.join(path_string, f"{recording_basename}_inter_spike_intervals.csv"), index=False)

    def extract_hd_tuning_parameters(self, data, path_string, recording_basename):
        """
        Extract head direction tuning parameters, cross-validate using halves and alternating bins.
        """
        # Load data
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['Wake'].intersect(position.time_support)

        feature = position['ry']

        # Compute tuning curves for the entire session
        tuning_curves_xr = nap.compute_tuning_curves(
            data=spikes,
            features=feature,
            epochs=wake_ep,
            bins=120,
            range=[(0, 2 * np.pi)],
        )
        tuning_curves = tuning_curves_xr.to_pandas()
        smooth_tuning_curves = self.smooth_angular_tuning_curves(tuning_curves)

        # Cross-validate with session halves
        tuning_curves_1st_half, tuning_curves_2nd_half, smooth_tuning_curves_1st_half, smooth_tuning_curves_2nd_half = self.cross_validate_tuning_curves(
            wake_ep, feature, spikes, position
        )

        # Cross-validate with alternating bins
        tuning_curves_odd_bins, tuning_curves_even_bins, smooth_tuning_curves_odd_bins, smooth_tuning_curves_even_bins = self.cross_validate_alternating_bins(
            wake_ep, feature, spikes, position
        )

        # Calculate Rayleigh properties
        mean_vector, mean_vector_length, R_value, preferred_direction = self.calculate_rayleigh_vector(smooth_tuning_curves)

        # Compute spatial information
        spatial_information = nap.compute_mutual_information(tuning_curves_xr)['bits/spike'].values

        # Compute explained variance for each neuron

        explained_variance = self.calculate_explained_variance(spikes, position, wake_ep, smooth_tuning_curves)

        # Save HD tuning properties
        pd.DataFrame({
            'mean_vector_real': np.real(mean_vector),
            'mean_vector_imag': np.imag(mean_vector),
            'mean_vector_length': mean_vector_length,
            'R_value': R_value,
            'preferred_direction': preferred_direction,
            'spatial_information': spatial_information.flatten(),
            'explained_variance': explained_variance,
        }).to_csv(os.path.join(path_string, f"{recording_basename}_HDTuning_Properties.csv"), index=False)

        # Save HD tuning curves
        for name, df in {
            'HDTuning_Curves': tuning_curves,
            'HDTuning_Curves_smooth': smooth_tuning_curves,
            'HDTuning_Curves_1st_half': tuning_curves_1st_half,
            'HDTuning_Curves_2nd_half': tuning_curves_2nd_half,
            'HDTuning_Curves_smooth_1st_half': smooth_tuning_curves_1st_half,
            'HDTuning_Curves_smooth_2nd_half': smooth_tuning_curves_2nd_half,
            'HDTuning_Curves_odd_bins': tuning_curves_odd_bins,
            'HDTuning_Curves_even_bins': tuning_curves_even_bins,
            'HDTuning_Curves_smooth_odd_bins': smooth_tuning_curves_odd_bins,
            'HDTuning_Curves_smooth_even_bins': smooth_tuning_curves_even_bins,
        }.items():
            df.to_csv(os.path.join(path_string, f"{recording_basename}_{name}.csv"))

        print(f"HD tuning properties and curves for {recording_basename} saved.")

    def extract_AHV_tuning_parameters(self, data, path_string, recording_basename):
        """Extracts and saves AHV tuning parameters, cross-validation results, and quadratic fit properties."""
        # Load position and spikes
        position = data.position
        spikes = data.spikes
        wake_ep = data.epochs['Wake'].intersect(position.time_support)

        # Compute Angular Head Velocity (AHV)
        timestamps = position.index.values
        head_direction = position['ry'].values  # Ensure correct column name
        dt = np.diff(timestamps)
        circular_diff_hd = np.angle(np.exp(1j * np.diff(head_direction)))
        ahv = circular_diff_hd / dt

        # Convert to Pynapple Tsd
        ahv_tsd = nap.Tsd(t=timestamps[1:], d=ahv)

        feature = ahv_tsd

        # Compute raw AHV tuning curves
        ahv_minmax = (-50 * np.pi / 180, 50 * np.pi / 180)
        tuning_curves = nap.compute_tuning_curves(
            data=spikes, features=feature, epochs=wake_ep, bins=30, range=[ahv_minmax]
        ).to_pandas()

        # Cross-validation
        tuning_curves_1st_half, tuning_curves_2nd_half, _, _ = self.cross_validate_tuning_curves(
            wake_ep, feature, spikes, position)
        tuning_curves_odd_bins, tuning_curves_even_bins, _, _ = self.cross_validate_alternating_bins(
            wake_ep, feature, spikes, position)

        # Fit quadratic models and compute asymmetry index
        ahv_bins = tuning_curves.index.values
        results = []
        
        for neuron in tuning_curves.columns:
            firing_rates = tuning_curves[neuron].values
            popt, _ = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, ahv_bins, firing_rates)
            a, b, c = popt

            pos_firing = sum(firing_rates[ahv_bins > 0])
            neg_firing = sum(firing_rates[ahv_bins < 0])
            asymmetry_index = (pos_firing - neg_firing) / (pos_firing + neg_firing)
            results.append([a, b, c, asymmetry_index])

        # Save data
        for name, df in {
            'AHV_Tuning_Curves': tuning_curves,
            'AHV_Tuning_Curves_1st_half': tuning_curves_1st_half,
            'AHV_Tuning_Curves_2nd_half': tuning_curves_2nd_half,
            'AHV_Tuning_Curves_odd_bins': tuning_curves_odd_bins,
            'AHV_Tuning_Curves_even_bins': tuning_curves_even_bins,
        }.items():
            df.to_csv(os.path.join(path_string, f"{recording_basename}_{name}.csv"))
        pd.DataFrame(results, columns=['a', 'b', 'c', 'asymmetry_index']).to_csv(
            os.path.join(path_string, f"{recording_basename}_AHV_fit.csv"), index=False)

    def extract_waveform_parameters(self, data, path_string, recording_basename):
        """Extract waveform parameters for all neurons."""
        print("To skip waveform extraction, simply comment out extract_waveform_parameters in config.yaml")

        mean_wf, max_ch = data.load_mean_waveforms()
        pd.concat(mean_wf, names=['neuron', 'time_s']).to_csv(
            os.path.join(path_string, f"{recording_basename}_mean_wf.csv"))
        pd.Series(max_ch, name='max_channel').to_csv(
            os.path.join(path_string, f"{recording_basename}_max_ch.csv"))

        max_wf = self.get_max_waveform(mean_wf, max_ch)
        trough_to_peaks = self.get_trough_to_peak(max_wf)

        pd.Series(trough_to_peaks, name='trough_to_peak').to_csv(
            os.path.join(path_string, f"{recording_basename}_waveform_parameters.csv"))

    # Utility Methods
    def calculate_mean_isi(self, spike_times):
        """Calculate the mean inter-spike interval."""
        spike_times = spike_times[spike_times > 0]
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            isis = isis[isis <= 0.5]
            return np.median(isis) if len(isis) > 0 else None
        return None

    def calculate_explained_variance(self, spikes, position, wake_ep, smooth_tuning_curves):
        explained_variances = []  # List to store explained variance for all neurons

        def find_firing_rate(direction, directions, firing_rates):
            """
            Find the firing rate corresponding to the closest head direction bin.

            Parameters:
                direction (float): The head direction in radians.
                directions (array-like): Array of head direction bin centers.
                firing_rates (array-like): Array of firing rates for each bin.

            Returns:
                float: The firing rate corresponding to the closest bin.
            """
            closest_idx = (np.abs(directions - direction)).argmin()
            return firing_rates[closest_idx]

        for neuron_id in spikes.keys():

            try:
                # Calculate spike rates
                wake_spikes = spikes[neuron_id].restrict(wake_ep)
                spike_rates = wake_spikes.count(bin_size=0.75, time_units='s') / 0.75

                # If spike rates are empty, append NaN and continue
                if spike_rates.values.size == 0:
                    print(f"No spikes detected for neuron {neuron_id}. Assigning NaN...")
                    explained_variances.append(np.nan)
                    continue

                # Bin the head direction
                head_direction_binned = position['ry'].bin_average(bin_size=0.75, ep=wake_ep, time_units='s')

                # Extract tuning curve
                tuning_curve = smooth_tuning_curves[neuron_id]
                directions = tuning_curve.index.values
                firing_rates = tuning_curve.values

                # Convert head direction to pandas Series
                head_direction_binned_series = pd.Series(
                    data=head_direction_binned.values,
                    index=head_direction_binned.index
                )

                # Map head directions to firing rates
                predicted_rates = head_direction_binned_series.apply(
                    lambda direction: find_firing_rate(direction, directions, firing_rates)
                )

                # If shapes of spike_rates and predicted_spike_rate don't match, append NaN
                if spike_rates.shape[0] != predicted_rates.shape[0]:
                    print(f"Mismatch in spike rates and predicted rates for neuron {neuron_id}. Assigning NaN...")
                    explained_variances.append(np.nan)
                    continue

                # Residuals
                residuals = spike_rates.values - predicted_rates.values

                # Variance calculations
                var_residuals = np.var(residuals)
                var_true = np.var(spike_rates.values)

                # If variance of true spike rates is zero, append NaN
                if var_true == 0:
                    print(f"Zero variance in true spike rates for neuron {neuron_id}. Assigning NaN...")
                    explained_variances.append(np.nan)
                    continue

                # Explained variance
                explained_variance = 1 - (var_residuals / var_true)
                explained_variances.append(explained_variance)

            except Exception as e:
                # Handle any unexpected errors
                print(f"Error processing neuron {neuron_id}: {e}. Assigning NaN...")
                explained_variances.append(np.nan)

        # Count neurons with NaN explained variance
        num_nan = np.isnan(explained_variances).sum()

        # Print the count if greater than 0
        if num_nan > 0:
            print(f"\nWarning: {num_nan} neurons have NaN explained variance.\n")

        return explained_variances


    def calculate_rayleigh_vector(self, tuning_curves):
        """Calculate Rayleigh vector properties for tuning curves."""
        complex_angles = tuning_curves.values * np.exp(1j * tuning_curves.index.to_numpy())[:, np.newaxis]
        mean_vector = np.mean(complex_angles, axis=0)
        mean_vector_length = np.abs(mean_vector)
        R_value = mean_vector_length / tuning_curves.shape[0]
        preferred_direction = np.angle(mean_vector)
        return mean_vector, mean_vector_length, R_value, preferred_direction

    def smooth_angular_tuning_curves(self, tuning_curves, window=20, deviation=3.0):
        """Smooth angular tuning curves using a Gaussian kernel."""
        new_tuning_curves = {}
        for i in tuning_curves.columns:
            tcurves = tuning_curves[i]
            offset = np.mean(np.diff(tcurves.index.values))
            padded = pd.Series(
                index=np.hstack((tcurves.index.values - (2 * np.pi) - offset,
                                 tcurves.index.values,
                                 tcurves.index.values + (2 * np.pi) + offset)),
                data=np.hstack((tcurves.values, tcurves.values, tcurves.values)),
            )
            smoothed = padded.rolling(window=window, win_type="gaussian", center=True, min_periods=1).mean(std=deviation)
            new_tuning_curves[i] = smoothed.loc[tcurves.index]
        return pd.DataFrame.from_dict(new_tuning_curves)

    def cross_validate_tuning_curves(self, wake_ep, feature, spikes, position):
        """
        Cross-validate tuning curves by splitting the wake epoch into two halves.
        """
        wake_ep_center = (wake_ep['end'] - wake_ep['start']) / 2
        sub_wake_ep_1 = nap.IntervalSet(
            start=wake_ep['start'], end=wake_ep['start'] + wake_ep_center, time_units="s"
        ).intersect(position.time_support)
        sub_wake_ep_2 = nap.IntervalSet(
            start=wake_ep['start'] + wake_ep_center, end=wake_ep['end'], time_units="s"
        ).intersect(position.time_support)

        tuning_curves_1 = nap.compute_tuning_curves(
            data=spikes,
            features=feature,
            epochs=sub_wake_ep_1,
            bins=120,
            range=[(0, 2 * np.pi)],
        ).to_pandas()
        tuning_curves_2 = nap.compute_tuning_curves(
            data=spikes,
            features=feature,
            epochs=sub_wake_ep_2,
            bins=120,
            range=[(0, 2 * np.pi)],
        ).to_pandas()

        smooth_tuning_curves_1 = self.smooth_angular_tuning_curves(tuning_curves_1)
        smooth_tuning_curves_2 = self.smooth_angular_tuning_curves(tuning_curves_2)

        return tuning_curves_1, tuning_curves_2, smooth_tuning_curves_1, smooth_tuning_curves_2

    def cross_validate_alternating_bins(self, wake_ep, feature, spikes, position, bin_duration=10):
        """
        Cross-validate tuning curves by splitting the wake epoch into alternating bins.
        """
        # Generate bins using Pynapple IntervalSet
        start_time = wake_ep.start[0]  # Access the first value directly
        end_time = wake_ep.end[-1]     # Access the last value directly

        bin_edges = np.arange(start_time, end_time, bin_duration)
        bin_intervals = nap.IntervalSet(start=bin_edges[:-1], end=bin_edges[1:], time_units="s")

        # Separate odd and even bins
        odd_bins = bin_intervals[np.arange(0, len(bin_intervals), 2)]
        even_bins = bin_intervals[np.arange(1, len(bin_intervals), 2)]

        # Restrict bins to the position's time support
        odd_bins = odd_bins.intersect(position.time_support)
        even_bins = even_bins.intersect(position.time_support)

        # Compute tuning curves for odd bins
        tuning_curves_odd = nap.compute_tuning_curves(
            data=spikes,
            features=feature,
            epochs=odd_bins,
            bins=120,
            range=[(0, 2 * np.pi)],
        ).to_pandas()

        # Compute tuning curves for even bins
        tuning_curves_even = nap.compute_tuning_curves(
            data=spikes,
            features=feature,
            epochs=even_bins,
            bins=120,
            range=[(0, 2 * np.pi)],
        ).to_pandas()

        # Smooth tuning curves
        smooth_tuning_curves_odd = self.smooth_angular_tuning_curves(tuning_curves_odd)
        smooth_tuning_curves_even = self.smooth_angular_tuning_curves(tuning_curves_even)

        return tuning_curves_odd, tuning_curves_even, smooth_tuning_curves_odd, smooth_tuning_curves_even

    def detect_oscillations(self, data, path_string, recording_basename):
        sws_ep = data.read_neuroscope_intervals('sws')

        metadata_files = [f for f in os.listdir(path_string) if f.endswith("_channel.txt")]
        if not metadata_files:
            print(f"  No oscillation channel files found for {recording_basename} — skipping.")
            return

        for metadata_file in metadata_files:
            filename_parts = metadata_file.split("_")
            if len(filename_parts) < 2 or not filename_parts[1].startswith("channel.txt"):
                continue  # skip control_channel.txt and malformed files

            oscillation_type = filename_parts[0].lower()

            try:
                with open(os.path.join(path_string, metadata_file), "r", encoding='utf-8-sig') as f:
                    channel = int(f.read().strip())
            except ValueError:
                print(f"  Invalid channel number in {metadata_file}. Skipping.")
                continue

            if oscillation_type == "ripple":
                params = {
                    "freq_band":          (100, 300),
                    "thres_band":         (4, 15),
                    "noise_thres_band":   (4, 10),
                    "duration_band":      (0.01, 0.1),
                    "min_inter_duration": 0.02,
                    "smoothing_bins":     40,
                    "evt_extension":      ".evt.py.rip",
                    "evt_abbreviation":   "rip",
                }
            # elif oscillation_type == "spindle":
            #     params = {
            #         "freq_band":          (10, 16),
            #         "thres_band":         (1, 10),
            #         "noise_thres_band":   (1, 7),
            #         "duration_band":      (0.4, 2.1),
            #         "min_inter_duration": 0.02,
            #         "smoothing_bins":     51,
            #         "evt_extension":      ".evt.py.spn",
            #         "evt_abbreviation":   "spn",
            #     }
            else:
                print(f"  Unsupported oscillation type: {oscillation_type}. Skipping.")
                continue

            print(f"  {oscillation_type.capitalize()} channel {channel} — running detection.")

            try:
                lfp = data.load_lfp(channel=channel, extension=".eeg")
            except Exception as e:
                print(f"  Error loading LFP for channel {channel}: {e}")
                continue

            # Control channel noise rejection
            denoised_ep = sws_ep
            control_channel_file = os.path.join(path_string, f"{oscillation_type}_control_channel.txt")
            if os.path.isfile(control_channel_file):
                try:
                    with open(control_channel_file, "r", encoding='utf-8-sig') as f:
                        control_channel = int(f.read().strip())
                    control_lfp = data.load_lfp(channel=control_channel, extension=".eeg")
                    noise_ep, _ = detect_oscillatory_events_hilbert(
                        control_lfp, sws_ep,
                        params["freq_band"], params["noise_thres_band"],
                        params["duration_band"], params["min_inter_duration"],
                        params["smoothing_bins"])
                    denoised_ep = sws_ep.set_diff(noise_ep)
                    print(f"  Control channel {control_channel}: {len(noise_ep)} noise epochs removed "
                          f"({(noise_ep['end'] - noise_ep['start']).sum():.1f} s).")
                except Exception as e:
                    print(f"  Control channel rejection failed: {e}. Proceeding without control.")

            # Detect events
            osc_ep, osc_tsd = detect_oscillatory_events_hilbert(
                lfp, denoised_ep,
                params["freq_band"], params["thres_band"],
                params["duration_band"], params["min_inter_duration"],
                params["smoothing_bins"])

            print(f"  Found {len(osc_ep)} {oscillation_type}s.")

            # Save CSV
            osc_ep.as_dataframe().to_csv(
                os.path.join(path_string, f"{recording_basename}_{oscillation_type}_ep.csv"))

            # Save .evt file for Neuroscope (start / peak / stop)
            starts = osc_ep.as_units('ms')['start'].values
            peaks  = osc_tsd.as_units('ms').index.values
            ends   = osc_ep.as_units('ms')['end'].values
            abbrev = params["evt_abbreviation"]

            datatowrite = np.vstack((starts, peaks, ends)).T.flatten()
            n = len(osc_ep)
            texttowrite = np.vstack((
                np.repeat(np.array([f'{abbrev} start 1']), n),
                np.repeat(np.array([f'{abbrev} peak 1']),  n),
                np.repeat(np.array([f'{abbrev} stop 1']),  n),
            )).T.flatten()

            evt_file = os.path.join(path_string, data.basename + params["evt_extension"])
            with open(evt_file, 'w') as f:
                for t, label in zip(datatowrite, texttowrite):
                    f.write(f"{t:1.6f}\t{label}\n")

            print(f"  Saved {evt_file}")

    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype='band')

    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """Apply a bandpass filter to the data."""
        b, a = self._butter_bandpass(lowcut, highcut, fs, order)
        return lfilter(b, a, data)

    def get_max_waveform(self, mean_wf, max_ch):
        """Get the waveform with the largest amplitude for each neuron."""
        max_wf = {key: mean_wf[key][max_ch[key]] for key in mean_wf.keys()}
        return max_wf

    def get_trough_to_peak(self, max_wf):
        """Calculate the trough-to-peak time for each neuron."""
        trough_to_peaks = {}
        for neuron, waveform in max_wf.items():
            trough = waveform.idxmin()
            peak = waveform.loc[trough:].idxmax()
            trough_to_peaks[neuron] = peak - trough
        return trough_to_peaks
