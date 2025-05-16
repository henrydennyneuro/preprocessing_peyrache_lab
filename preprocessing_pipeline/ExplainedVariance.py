import os
import pickle
import yaml
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt
from pathlib import Path

def smooth_angular_tuning_curves(tuning_curves, window=20, deviation=3.0):
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

directory = r"E:/B3200/B3208/B3208-240612"

data = ntm.load_session(directory, "neurosuite")

spikes = data.spikes
position = data.position
wake_ep = data.epochs['Wake'].intersect(position.time_support)

tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes,
    feature=position['ry'],
    ep=wake_ep,
    nb_bins=120,
    minmax=(0, 2 * np.pi)
)

smooth_tuning_curves = smooth_angular_tuning_curves(tuning_curves)

# Initialize a list to store explained variances
explained_variances = []

# Define bin size
bin_size = 0.75  # Adjust the bin size in seconds (e.g., 0.1 for 100 ms)

# Iterate over all neurons in the dataset
for neuron_id in spikes.keys():

    # Calculate spike rates for the neuron
    wake_spikes = spikes[neuron_id].restrict(wake_ep)
    spike_rates = wake_spikes.count(bin_size=bin_size, time_units='s') / bin_size

    # Bin the head direction
    head_direction_binned = position['ry'].bin_average(bin_size=bin_size, ep=wake_ep, time_units='s')

    # Extract the tuning curve for the neuron
    tuning_curve = smooth_tuning_curves[neuron_id]

    # Extract directions and firing rates from the tuning curve
    directions = tuning_curve.index.values  # Index contains head direction bins (in radians)
    firing_rates = tuning_curve.values      # Values contain corresponding firing rates

    # Convert head_direction_binned to pandas Series
    head_direction_binned_series = pd.Series(
        data=head_direction_binned.values,
        index=head_direction_binned.index
    )

    # Function to find the closest direction and return the corresponding firing rate
    def find_firing_rate(direction, directions, firing_rates):
        closest_idx = (np.abs(directions - direction)).argmin()  # Find the closest bin index
        return firing_rates[closest_idx]

    # Map each head direction to its closest firing rate
    predicted_rates = head_direction_binned_series.apply(
        lambda direction: find_firing_rate(direction, directions, firing_rates)
    )

    # Create the predicted spike rate DataFrame
    predicted_spike_rate = pd.DataFrame({
        'time': head_direction_binned_series.index,
        'rate': predicted_rates.values
    })

    # Step 1: Residuals (Error Term)
    residuals = spike_rates.values - predicted_spike_rate['rate'].values

    # Step 2: Variance of Residuals
    var_residuals = np.var(residuals)

    # Step 3: Variance of True Firing Rates
    var_true = np.var(spike_rates.values)

    # Step 4: Calculate Explained Variance
    explained_variance = 1 - (var_residuals / var_true)

    # Append the result to the list
    explained_variances.append(explained_variance)

# Print all explained variances
print(explained_variances)
