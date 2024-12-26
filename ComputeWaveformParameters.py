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

def GetMaxWaveform(mean_wf, max_ch):

	"""
		First we have to get the waveform with the biggest amplitude for each spike. max_ch is a pandas dataframe where each
		row lists the channel with the greatest amplitude for each spike. mean_wf is a dictionary, where each key corresponds
		to a cell, with a dataframe of the average spike for each channel. 

		So we will create a new dictionary containing all the largest amplitude mean spikes for each neuron. Start by
		initialising the dictionary with the keys from the mean_wf dictionary
	"""

	max_wf = dict.fromkeys(list(mean_wf.keys()))

	"""
		Loop through the max_ch dataframe using the max channel value to select out the max waveform from the mean_wf
		dataframes nested in the dictionary.
	"""

	for keys in max_wf.keys():
		max_wf[keys] = mean_wf[keys][max_ch[keys]]

	return max_wf

def GetTroughToPeak(max_wf):

	"""
		Now we have the largest mean waveform for each neuron, we can 
	"""

	troughs = dict.fromkeys(list(max_wf.keys()))
	peaks = dict.fromkeys(list(max_wf.keys()))
	trough_to_peaks = dict.fromkeys(list(max_wf.keys()))

	shortened_waveform_container = dict.fromkeys(list(max_wf.keys()))
	slope_derivatives = dict.fromkeys(list(max_wf.keys()))

	for keys in max_wf:
		troughs[keys] = max_wf[keys].idxmin()
		
		shortened_waveform_container[keys] = max_wf[keys].loc[(troughs[keys]):]
		#slope_derivatives[keys] = shortened_waveform_container[keys].diff() I found using the 1st derivative was no improvement on max
		peaks[keys] = shortened_waveform_container[keys].idxmax()
		
		trough_and_peak_container = [troughs[keys], peaks[keys]]
		trough_to_peaks[keys] = trough_and_peak_container[1] - trough_and_peak_container[0]

	"""
		The plot below can be used to check how well troughs and peaks are detected for each cell. Troughs are easy to detect
		but peaks can be a little harder. I tried both just taking the max value after the trough, or the point at which the 
		first derivative was smallest (indicating a pleateau). I found taking the max was the most reliable.
	"""    

	# fig, axs = plt.subplots(10, 2)
	# for i, ax in enumerate(axs.flatten()):
	# 	ax.plot(max_wf[i])
	# 	ax.plot(shortened_waveform_container[i], color = "red")
	# 	ax.axvline(troughs[i], color = "black")
	# 	ax.axvline(peaks[i], color = "black")        
	# plt.show()


	return trough_to_peaks

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

		print(f'Running extraction for {recording_basename}')

		"""
			Search for previously created wavform files. Load the data from the specified session and extract task-relevant 
			variables.
		"""    

		waveform_filename = f'{directory}\\{recording_basename}_mean_wf.pkl'
		max_ch_filename = f'{directory}\\{recording_basename}_max_ch.pkl'
		
		file_exist = os.path.isfile(waveform_filename)

		if file_exist == True:

			print("Waveform file found, loading pickle files.")

			waveform_file = open(waveform_filename, 'rb')
			max_ch_file = open(max_ch_filename, 'rb')

			mean_wf = pickle.load(waveform_file)
			max_ch = pickle.load(max_ch_file)

		elif file_exist == False:

			print("Waveform files not found, loading and saving waveforms from neuroscope.")

			data = ntm.load_session(directory, "neurosuite")

			mean_wf, max_ch = data.load_mean_waveforms()

			with open(os.path.join(path_string, data.basename + "_mean_wf.pkl"), 'wb') as file:
				pickle.dump(mean_wf, file)

			with open(os.path.join(path_string, data.basename + "_max_ch.pkl"), 'wb') as file:
				pickle.dump(max_ch, file)

			# fig, axs = plt.subplots(10, 2)
			# for i, ax in enumerate(axs.flatten()):
			# 	for channels in mean_wf[i]:
			# 		ax.plot(mean_wf[i][channels])
			# plt.show()

		"""
			Find the channel with the largest spike for each cell, and use that spike to extract waveform parameters including:
				- Trough to Peak time
		"""    

		max_wf = GetMaxWaveform(mean_wf, max_ch)

		trough_to_peaks = GetTroughToPeak(max_wf)

		with open(os.path.join(path_string, recording_basename + "_waveform_parameters.pkl"), 'wb') as file:
			pickle.dump(trough_to_peaks, file)