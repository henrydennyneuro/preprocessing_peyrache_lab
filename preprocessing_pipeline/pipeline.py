import os
import yaml
import pickle
import json
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pathlib import Path
from scipy.signal import butter, lfilter, filtfilt
from scipy.optimize import curve_fit

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
        return files, steps

    def process_from_yaml(self, config_path):
        """Process files and steps specified in a YAML configuration."""
        files, steps = self.load_yaml_config(config_path)
        for file in files:
            full_path = Path(file)
            if full_path.exists():
                print(f"Processing file: {full_path}")
                self.process_recording(directory=full_path, steps=steps)
            else:
                print(f"File not found: {full_path}")
                
    def process_recording(self, directory, steps):
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

        with open(os.path.join(path_string, f"{recording_basename}_inter_spike_intervals.pkl"), 'wb') as file:
            pickle.dump([median_isis, median_sleep_isis], file)

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
        tuning_curves = nap.compute_1d_tuning_curves(
            group=spikes,
            feature=feature,
            ep=wake_ep,
            nb_bins=120,
            minmax=(0, 2 * np.pi)
        )
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
        spatial_information = nap.compute_1d_mutual_info(smooth_tuning_curves, spikes.restrict(wake_ep).to_tsd()).to_numpy()

        # Compute explained variance for each neuron

        explained_variance = self.calculate_explained_variance(spikes, position, wake_ep, smooth_tuning_curves)

        # Save HD tuning properties
        with open(os.path.join(path_string, f"{recording_basename}_HDtuning_properties.pkl"), 'wb') as file:
            pickle.dump([mean_vector, mean_vector_length, R_value, preferred_direction, spatial_information, explained_variance], file)

        # Save HD tuning curves
        with open(os.path.join(path_string, f"{recording_basename}_HDtuning_curves.pkl"), 'wb') as file:
            pickle.dump([
                tuning_curves,
                smooth_tuning_curves,
                tuning_curves_1st_half,
                tuning_curves_2nd_half,
                smooth_tuning_curves_1st_half,
                smooth_tuning_curves_2nd_half,
                tuning_curves_odd_bins,
                tuning_curves_even_bins,
                smooth_tuning_curves_odd_bins,
                smooth_tuning_curves_even_bins
            ], file)

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
        tuning_curves = nap.compute_1d_tuning_curves(
            group=spikes, feature=feature, ep=wake_ep, nb_bins=30, minmax=ahv_minmax
        )

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
        with open(os.path.join(path_string, f"{recording_basename}_AHV_tuning.pkl"), 'wb') as file:
            pickle.dump([tuning_curves, tuning_curves_1st_half, tuning_curves_2nd_half, 
                         tuning_curves_odd_bins, tuning_curves_even_bins], file)
        with open(os.path.join(path_string, f"{recording_basename}_AHV_fit.pkl"), 'wb') as file:
            pickle.dump(results, file)

    def detect_oscillatory_events(self, lfp, epoch, freq_band, thres_band, duration_band, min_inter_duration):
        """Detect oscillatory events in the LFP."""
        lfp = lfp.restrict(epoch)
        signal = self.bandpass_filter(lfp.as_units('s').values, freq_band[0], freq_band[1], lfp.rate)
        squared_signal = np.square(signal)  # Direct access to the array
        window = np.ones(51) / 51
        filtered_signal = filtfilt(window, 1, squared_signal)
        normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

        nSS = nap.Tsd(
            t=lfp.as_units('s').index.values, 
            d=normalized_signal, 
            time_support=epoch
        )
        osc_ep = nSS.threshold(thres_band[0], method='above').threshold(thres_band[1], method='below').time_support
        osc_ep = osc_ep.drop_short_intervals(duration_band[0], time_units='s').drop_long_intervals(duration_band[1], time_units='s')
        osc_ep = osc_ep.merge_close_intervals(min_inter_duration, time_units='s')

        return osc_ep, nSS

    def extract_waveform_parameters(self, data, path_string, recording_basename):
        """Extract waveform parameters for all neurons."""
        waveform_file = os.path.join(path_string, f"{recording_basename}_mean_wf.pkl")
        max_ch_file = os.path.join(path_string, f"{recording_basename}_max_ch.pkl")

        if os.path.isfile(waveform_file):
            with open(waveform_file, 'rb') as wf, open(max_ch_file, 'rb') as mc:
                mean_wf = pickle.load(wf)
                max_ch = pickle.load(mc)
        else:
            mean_wf, max_ch = data.load_mean_waveforms()
            with open(waveform_file, 'wb') as wf, open(max_ch_file, 'wb') as mc:
                pickle.dump(mean_wf, wf)
                pickle.dump(max_ch, mc)

        max_wf = self.get_max_waveform(mean_wf, max_ch)
        trough_to_peaks = self.get_trough_to_peak(max_wf)

        with open(os.path.join(path_string, f"{recording_basename}_waveform_parameters.pkl"), 'wb') as file:
            pickle.dump(trough_to_peaks, file)

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

        tuning_curves_1 = nap.compute_1d_tuning_curves(
            group=spikes,
            feature=feature,
            ep=sub_wake_ep_1,
            nb_bins=120,
            minmax=(0, 2 * np.pi),
        )
        tuning_curves_2 = nap.compute_1d_tuning_curves(
            group=spikes,
            feature=feature,
            ep=sub_wake_ep_2,
            nb_bins=120,
            minmax=(0, 2 * np.pi),
        )

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
        tuning_curves_odd = nap.compute_1d_tuning_curves(
            group=spikes,
            feature=feature,
            ep=odd_bins,
            nb_bins=120,
            minmax=(0, 2 * np.pi),
        )

        # Compute tuning curves for even bins
        tuning_curves_even = nap.compute_1d_tuning_curves(
            group=spikes,
            feature=feature,
            ep=even_bins,
            nb_bins=120,
            minmax=(0, 2 * np.pi),
        )

        # Smooth tuning curves
        smooth_tuning_curves_odd = self.smooth_angular_tuning_curves(tuning_curves_odd)
        smooth_tuning_curves_even = self.smooth_angular_tuning_curves(tuning_curves_even)

        return tuning_curves_odd, tuning_curves_even, smooth_tuning_curves_odd, smooth_tuning_curves_even

    def detect_oscillatory_events(self, data_path, epoch):
        """
        Detect oscillatory events in the LFP using metadata files for oscillation-specific analysis.
        
        Parameters:
        - data_path: Path to the data folder containing metadata and LFP files.
        - epoch: Epoch to restrict the LFP.
        """
        # Step 1: Check for .txt metadata files
        metadata_files = [f for f in os.listdir(data_path) if f.endswith("_channel.txt")]
        if not metadata_files:
            print(f"No metadata files found in {data_path}. Skipping oscillation detection.")
            return

        # Step 2: Iterate through metadata files for each oscillation type
        for metadata_file in metadata_files:
            # Parse the oscillation type from the metadata filename
            filename_parts = metadata_file.split("_")
            if len(filename_parts) < 2 or not filename_parts[1].startswith("channel.txt"):
                print(f"Invalid metadata file format: {metadata_file}. Skipping.")
                continue

            oscillation_type = filename_parts[0].lower()  # Extract oscillation type (e.g., "ripple", "spindle")
            print(f"Processing oscillation type: {oscillation_type}")

            # Load the channel number from the metadata file
            metadata_path = os.path.join(data_path, metadata_file)
            try:
                with open(metadata_path, "r") as file:
                    channel = int(file.read().strip())
            except ValueError:
                print(f"Invalid channel number in {metadata_file}. Skipping this file.")
                continue

            # Step 3: Load analysis parameters based on oscillation type
            if oscillation_type == "ripple":
                params = {
                    "freq_band": (100, 300),
                    "thres_band": (7, 10),
                    "duration_band": (0.01, 0.1),
                    "min_inter_duration": 0.02,
                }
            elif oscillation_type == "spindle":
                params = {
                    "freq_band": (10, 16),
                    "thres_band": (0.25, 20),
                    "duration_band": (0.4, 2.1),
                    "min_inter_duration": 0.02,
                }
            else:
                print(f"Unsupported oscillation type: {oscillation_type}. Skipping this file.")
                continue

            # Step 4: Load the LFP for the specified channel
            try:
                lfp = data.load_lfp(channel=channel, extension=".eeg")
                lfp = lfp.restrict(epoch)
            except Exception as e:
                print(f"Error loading LFP for channel {channel} in {metadata_file}: {e}")
                continue

            # Step 5: Perform oscillation detection
            print(f"Performing {oscillation_type} analysis on channel {channel}...")
            signal = self.bandpass_filter(lfp.as_units("s").values, params["freq_band"][0], params["freq_band"][1], lfp.rate)
            squared_signal = np.square(signal)
            window = np.ones(51) / 51
            filtered_signal = filtfilt(window, 1, squared_signal)
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

            nSS = nap.Tsd(
                t=lfp.as_units("s").index.values,
                d=normalized_signal,
                time_support=epoch
            )
            osc_ep = nSS.threshold(params["thres_band"][0], method="above").threshold(params["thres_band"][1], method="below").time_support
            osc_ep = osc_ep.drop_short_intervals(params["duration_band"][0], time_units="s").drop_long_intervals(params["duration_band"][1], time_units="s")
            osc_ep = osc_ep.merge_close_intervals(params["min_inter_duration"], time_units="s")

            # Save results
            basename = os.path.basename(data_path)
            osc_ep.save(os.path.join(data_path, f"{basename}_{oscillation_type}_ep"))
            print(f"Saved {oscillation_type} intervals for channel {channel}.")

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
