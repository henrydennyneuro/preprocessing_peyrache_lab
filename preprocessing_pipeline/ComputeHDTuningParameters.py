import os
import json
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt

from pathlib import Path

def calculate_rayleigh_vector(tuning_curves):
	# Convert angles to complex numbers
	complex_angles = tuning_curves.values * np.exp(1j * tuning_curves.index.to_numpy())[:, np.newaxis]

	# Calculate the mean vector
	mean_vector = np.mean(complex_angles, axis=0)

	# Calculate the length of the mean vector
	mean_vector_length = np.abs(mean_vector)

	# Calculate the Rayleigh R-value
	R_value = mean_vector_length / tuning_curves.shape[0]

	# Calculate the preferred direction (angle)
	preferred_direction = np.angle(mean_vector)

	return mean_vector, mean_vector_length, R_value, preferred_direction

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
	
	new_tuning_curves = {}  

	for i in tuning_curves.columns:
		tcurves = tuning_curves[i]
		offset = np.mean(np.diff(tcurves.index.values))
		padded  = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi)-offset,
												tcurves.index.values,
												tcurves.index.values+(2*np.pi)+offset)),
							data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
		smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)      
		new_tuning_curves[i] = smoothed.loc[tcurves.index]

	new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

	return new_tuning_curves

def CrossValidateTuningCurves(wake_ep, position, spikes):

    # In order to cross validate our tuning curves, we need to check the 
    # stability of the tuning curve across both halves of the epoch. The
    # following code subdivides the wake epoch, allowing us to plot tuning
    # curves for each half.

    wake_ep_center = (wake_ep['end']-wake_ep['start'])/2
    sub_wake_ep_1 = nap.IntervalSet(start=wake_ep['start'], \
                         end=wake_ep['start']+wake_ep_center, time_units='s')
    sub_wake_ep_1.intersect(position.time_support)

    sub_wake_ep_2 = nap.IntervalSet(start=wake_ep['start']+wake_ep_center, \
                         end=wake_ep['end'], time_units='s')
    sub_wake_ep_2.intersect(position.time_support)

    # wake_ep_quarter = (wake_ep['end']-wake_ep['start'])/4
    # sub_wake_ep_1 = nap.IntervalSet(start=wake_ep['start'], \
    #                     end=wake_ep['start']+wake_ep_quarter, time_units='s')
    # sub_wake_ep_1.intersect(position.time_support)

    # sub_wake_ep_2 = nap.IntervalSet(start=wake_ep['start']+wake_ep_quarter, \
    #                     end=wake_ep['start']+(wake_ep_quarter*2), time_units='s')
    # sub_wake_ep_2.intersect(position.time_support)
    
    # COMPUTING TUNING CURVES FOR FIRST HALF OF WAKE EPOCH
    tuning_curves_1 = nap.compute_tuning_curves(
        data=spikes, features=position['ry'], epochs=sub_wake_ep_1,
        bins=120, range=[(0, 2*np.pi)],
    ).to_pandas()

    # COMPUTING TUNING CURVES FOR SECOND HALF OF WAKE EPOCH
    tuning_curves_2 = nap.compute_tuning_curves(
        data=spikes, features=position['ry'], epochs=sub_wake_ep_2,
        bins=120, range=[(0, 2*np.pi)],
    ).to_pandas()
    
    # SMOOTH TUNING CURVES TO IMPROVE LEGIBILITY
    smooth_tuning_curves_1 = smoothAngularTuningCurves(tuning_curves_1)
    smooth_tuning_curves_2 = smoothAngularTuningCurves(tuning_curves_2)

    return tuning_curves_1, tuning_curves_2, smooth_tuning_curves_1, smooth_tuning_curves_2

if __name__ == "__main__":


	"""
		This script calculates the Rayleigh r-value for each neuron. R-vector is a way to quantify the 
		directionality or preferred direction of firing rates in neural spike data.

		Set up path for loading files and saving metadata.
	"""    
	
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

		print(f'Computing tuning properties for {recording_basename}')

		"""
			Load spiking data from session.
		"""    
		data = ntm.load_session(directory, "neurosuite")

		spikes = data.spikes
		position = data.position
		wake_ep = data.epochs['Wake'].intersect(position.time_support)

		"""
			Compute tuning curves for each neuron. 
		"""

		tuning_curves_xr = nap.compute_tuning_curves(
			data=spikes, features=position['ry'], epochs=wake_ep,
			bins=120, range=[(0, 2*np.pi)],
		)
		tuning_curves = tuning_curves_xr.to_pandas()

		smooth_tuning_curves = smoothAngularTuningCurves(tuning_curves)

		"""
			Cross validate tuning curves to check that any tuning is stable. 
		"""
		tuning_curves_1st_half, tuning_curves_2nd_half, smooth_tuning_curves_1st_half, smooth_tuning_curves_2nd_half \
			= CrossValidateTuningCurves(wake_ep, position, spikes)

		"""
			Compute rayleigh HD properties. 
		"""

		mean_vector, mean_vector_length, R_value, preferred_direction = calculate_rayleigh_vector(smooth_tuning_curves)

		"""
			Compute spatial information of each tuning curve. 
		"""
		
		spatial_information = nap.compute_mutual_information(tuning_curves_xr)['bits/spike'].values

		spatial_information_as_ndarray = list(spatial_information)

		"""
			If necissary, you can plot all the tuning curves and display the calculated preferred orientation. 
		"""

		# plt.figure()
		# for count, value in enumerate(spikes): #filtered_spikes
		#     plt.subplot(10,10,count+1, projection = 'polar')
		#     plt.plot(smooth_tuning_curves[value]) #filtered_spikes
		#     plt.axvline(preferred_direction[value], color='black', linewidth=2)
		#     plt.xticks([])
		#     plt.ylabel('')
		#     plt.text(1,1,str(round(spatial_information_as_ndarray[value], 3)), color = 'red')
		# plt.show()

		"""
			Save HD properties as pickle. 
		"""

		pd.DataFrame({
			'mean_vector_real': np.real(mean_vector),
			'mean_vector_imag': np.imag(mean_vector),
			'mean_vector_length': mean_vector_length,
			'R_value': R_value,
			'preferred_direction': preferred_direction,
			'spatial_information': spatial_information_as_ndarray,
		}).to_csv(os.path.join(path_string, f'{recording_basename}_HDTuning_Properties.csv'), index=False)

		for name, df in {
			'HDTuning_Curves': tuning_curves,
			'HDTuning_Curves_smooth': smooth_tuning_curves,
			'HDTuning_Curves_1st_half': tuning_curves_1st_half,
			'HDTuning_Curves_2nd_half': tuning_curves_2nd_half,
			'HDTuning_Curves_smooth_1st_half': smooth_tuning_curves_1st_half,
			'HDTuning_Curves_smooth_2nd_half': smooth_tuning_curves_2nd_half,
		}.items():
			df.to_csv(os.path.join(path_string, f'{recording_basename}_{name}.csv'))

		print(f'Computation for {recording_basename} complete')