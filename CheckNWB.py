import os
import sys
import pickle
import pynapple as nap
import nwbmatic as ntm
import yaml

from pathlib import Path

# Usage: python CheckNWB.py [config.yaml]   (defaults to recordings_for_adel.yaml)
yaml_file = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\hdenny\Documents\Toolbox\preprocessing_peyrache_lab\configs\recordings_for_adel.yaml"
with open(yaml_file, "r") as file:
    config = yaml.safe_load(file)

# Extract directory and file list
data_directory = config.get("data_directory", "")
datasets = config.get("files", [])

# Check if datasets exist
if not datasets:
    print("No datasets found in the YAML file. Exiting.")
    exit()

for dataset in datasets:
    directory = os.path.join(data_directory, dataset)
    path_string = Path(directory)
    recording_basename = os.path.basename(directory)

    print(f"Processing recording: {recording_basename}")

    # Load session data
    data = ntm.load_session(directory, "neurosuite")

    # spikes = data.spikes

    # # Load waveform parameters
    # waveform_parameter_filename = f"{directory}/{recording_basename}_waveform_parameters.pkl"
    # with open(waveform_parameter_filename, "rb") as waveform_file:
    #     trough_to_peaks = pickle.load(waveform_file)

    # if len(spikes) == len(trough_to_peaks):
    #     print("Same number of t2p's as spikes")
    # else:
    #     print(f"Warning! {recording_basename} has uneven spikes ({len(spikes)}) to t2p's ({len(trough_to_peaks)})")