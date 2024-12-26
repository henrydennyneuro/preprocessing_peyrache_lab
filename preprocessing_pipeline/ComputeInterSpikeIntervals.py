import os
import pickle
import json
import numpy as np
import pandas as pd
import pickle as pickle  
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt

from pathlib import Path

def CalculateMeanISI(neuron_spikes):
    """
    Calculate the median inter-spike interval for a single neuron,
    excluding ISIs longer than 1 second.
    
    Parameters:
    neuron_spikes (ndarray): A 1D numpy array containing the spike times of a neuron.
    
    Returns:
    median_isi (float): The median ISI for the neuron. Returns None if ISI cannot be calculated.
    """
    neuron_spikes = neuron_spikes[neuron_spikes > 0]  # Ignore non-spike entries (e.g., 0s or negative values)
    if len(neuron_spikes) > 1:
        isis = np.diff(neuron_spikes)
        isis = isis[isis <= 0.5]  # Exclude ISIs longer than 0.5 seconds
        if len(isis) > 0:
            median_isi = np.median(isis)
        else:
            median_isi = None  # No valid ISIs to calculate median
    else:
        median_isi = None  # If there's only one spike or none, ISI cannot be calculated
    
    return median_isi


if __name__ == '__main__':

	what_source = input("Run on whole dataset? (y/n):")

	if what_source == "y":
		open_dataset = open("B2904.json")
		dataset = json.load(open_dataset)

		dataset_number = np.arange(len(dataset[1]['datasets'])).tolist()
	else:
		directory = input("Enter folder directory: ")

		dataset_number = [0]

	for datasets in dataset_number:

		if what_source == 'y':
			directory = dataset[0]['data_directory'] + dataset[1]['datasets'][dataset_number[datasets]]

		path_string = Path(directory)
		recording_basename = os.path.basename(directory)

		print(f'Running ISI extraction for {recording_basename}')

		data = ntm.load_session(directory, "neurosuite")

		spikes = data.spikes

		sleep_ep = data.epochs['Sleep']
		sleep_spikes = spikes.restrict(sleep_ep)

		median_isis_list = []
		median_sleep_isis_list = []

		for neuron in spikes:
			median_isis_list.append(CalculateMeanISI(spikes[neuron].times()))
			median_sleep_isis_list.append(CalculateMeanISI(sleep_spikes[neuron].times()))

		with open(os.path.join(path_string, recording_basename + "_inter_spike_intervals.pkl"), 'wb') as file:
			pickle.dump([median_isis_list, median_sleep_isis_list], file)

		print(f"ISI's for {recording_basename} extracted and saved.")

