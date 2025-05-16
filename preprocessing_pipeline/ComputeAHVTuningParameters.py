import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import uniform_filter1d

# Load YAML configuration file
yaml_file = r"C:\Users\Henry Denny\TRNPhysiology\configs\TRN_project.yaml"
with open(yaml_file, "r") as file:
    config = yaml.safe_load(file)

# Extract directory and file list
data_directory = config.get("data_directory", "")
datasets = config.get("files", [])

# Check if datasets exist
if not datasets:
    print("No datasets found in the YAML file. Exiting.")
    exit()

# Process the first dataset (modify to loop through all datasets if needed)
dataset = datasets[0]  # Select the first dataset
directory = os.path.join(data_directory, dataset)
path_string = Path(directory)
recording_basename = os.path.basename(directory)

print(f"Processing recording: {recording_basename}")

# Load session data
data = nap.load_session(directory, "neurosuite")  # Adjust the loader as needed for your format

# Extract position and spikes data
position = data.position  # Ensure this is a nap.TsdFrame
spikes = data.spikes      # Ensure this is a nap.TsGroup

# Define the epoch of interest (e.g., waking state)
wake_ep = data.epochs["Wake"]  # Replace "wake" with the appropriate key for your wake epochs

# Extract head direction (in radians)
head_direction = position["ry"].values  # Replace 'ry' with the correct column for head direction in radians

# Extract timestamps
timestamps = position.index.values  # Time in seconds

# Compute Angular Head Velocity (AHV)
# Compute AHV with circular handling
dt = np.diff(timestamps)  # Time intervals
circular_diff_hd = np.angle(np.exp(1j * np.diff(head_direction)))  # Circular difference in radians
ahv = circular_diff_hd / dt  # Convert to angular velocity (radians/second)

# Define AHV range in radians per second
ahv_minmax = (-50 * np.pi / 180, 50 * np.pi / 180)  # [-0.873, +0.873] rad/s

# Create a Pynapple Tsd object for the unbinned AHV
ahv_tsd = nap.Tsd(t=timestamps[1:], d=ahv)  # Unbinned AHV

# Compute AHV tuning curves directly from the raw AHV
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes,               # Group of spikes
    feature=ahv_tsd,            # Unbinned AHV
    ep=wake_ep,                 # Epoch of interest (e.g., waking state)
    nb_bins=30,                 # Number of bins for AHV
    minmax=ahv_minmax           # Range in radians per second
)

# Plot tuning curves
import matplotlib.pyplot as plt

neurons = list(tuning_curves.keys())
num_neurons = len(neurons)

# Determine optimized layout for subplots
def calculate_subplot_layout(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

rows, cols = calculate_subplot_layout(num_neurons)

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()  # Flatten in case of multidimensional array

# Plot each tuning curve in its subplot
for i, neuron in enumerate(neurons):
    ax = axes[i]
    ax.plot(tuning_curves[neuron])
    ax.set_title(f"Neuron {neuron}")
    ax.set_xlabel("AHV (rad/s)")
    ax.set_ylabel("Firing rate (Hz)")

# Hide unused subplots
for ax in axes[len(neurons):]:
    ax.axis("off")

plt.tight_layout()
plt.show()


#######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define quadratic function for fitting
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# AHV bins (assuming they are the index)
ahv_bins = tuning_curves.index.values

# Storage for extracted features
results = []

for neuron in tuning_curves.columns:
    firing_rates = tuning_curves[neuron].values

    # Fit quadratic function
    popt, _ = curve_fit(quadratic, ahv_bins, firing_rates)
    a, b, c = popt

    # Compute peak and minimum locations
    fitted_curve = quadratic(ahv_bins, *popt)
    peak_index = np.argmax(fitted_curve)
    min_index = np.argmin(fitted_curve)
    peak_ahv = ahv_bins[peak_index]
    min_ahv = ahv_bins[min_index]

    # Compute asymmetry index
    pos_ahv = ahv_bins[ahv_bins > 0]
    neg_ahv = ahv_bins[ahv_bins < 0]
    pos_firing = fitted_curve[ahv_bins > 0].sum()
    neg_firing = fitted_curve[ahv_bins < 0].sum()
    asymmetry_index = (pos_firing - neg_firing) / (pos_firing + neg_firing)

    # Determine tuning type based on 'a' coefficient
    if a > 0:
        tuning_type = "Biphasic (U-Shaped)"
    elif a < 0:
        tuning_type = "Inverted U-Shaped"
    else:
        tuning_type = "Other"

    results.append({
        "Neuron": neuron,
        "a (quad fit)": a,
        "b (quad fit)": b,
        "c (quad fit)": c,
        "Peak AHV": peak_ahv,
        "Min AHV": min_ahv,
        "Asymmetry Index": asymmetry_index,
        "Tuning Type": tuning_type
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv("AHV_tuning_analysis.csv", index=False)

# Display extracted parameters
print(results_df)

# Plot example neuron fits
num_neurons = len(tuning_curves.columns)
rows, cols = calculate_subplot_layout(num_neurons)

# Create figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()  # Flatten for easy iteration

# Plot each neuron’s tuning curve with quadratic fit
for i, neuron in enumerate(tuning_curves.columns):
    firing_rates = tuning_curves[neuron].values
    popt, _ = curve_fit(quadratic, ahv_bins, firing_rates)
    
    # Generate smooth AHV range for plotting
    smooth_ahv = np.linspace(ahv_bins.min(), ahv_bins.max(), 100)
    fitted_curve = quadratic(smooth_ahv, *popt)

    # Plot original tuning and fit
    axes[i].plot(ahv_bins, firing_rates, 'o', markersize=3, label="Data")
    axes[i].plot(smooth_ahv, fitted_curve, 'r-', label="Quadratic Fit")
    axes[i].set_title(f"Neuron {neuron}")
    axes[i].set_xlabel("AHV (rad/s)")
    axes[i].set_ylabel("Firing rate (Hz)")
    # axes[i].legend(fontsize=8)

# Hide any unused subplots (if neuron count isn’t a perfect square)
for ax in axes[num_neurons:]:
    ax.axis("off")

plt.tight_layout()
plt.show()