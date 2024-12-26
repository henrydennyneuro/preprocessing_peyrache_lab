import os
import pickle
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import pynacollada as pyna
import matplotlib.pyplot as plt

from pathlib import Path
from ValidateTTLs import extractTTL, getEpochDuration

if __name__ == '__main__':

    """
        This script is designed to smooth out periods where TTL detection was interrupted and save the IntervalSet as a pcikle
        file. 

        For TTL extraction, you must give the exact analogin filename where the TTL was recorded. Giving the folder only will
        not work, and will crash the script.

        First we will extract all the filenaming parameters we need to save the IntervalSet
    """    
    analogin_file = input("Enter analogin dat file location: ")

    file_location = os.path.dirname(analogin_file)
    path_string = Path(analogin_file)
    path_parts = path_string.parts
    recording_basename = path_parts[4]
    
    """
        Extract TTL and convert it into a pynapple timeseries to make it easier to work with. We will use the pynapple threshold
        function to find the TTL voltage peaks and the time_support method to find consecutive supra-threshold values. We then 
        record the start/end timestamps of both the troughs and the peaks. This will allow us to check TTL's are stable and that
        no TTL's are dropped.

        Load in epochs from NWB so we can align the stim epochs start times to the concatenated dat file
    """    

    ttl = extractTTL(analogin_file)

    ttl_tsd = nap.Tsd(ttl)

    ttl_threshold = 30000
    detected_peaks = ttl_tsd.threshold(ttl_threshold)
    peaks_epochs = detected_peaks.time_support
    troughs_epochs = ttl_tsd.time_support.set_diff(peaks_epochs)


    data = ntm.load_session(file_location, "neurosuite")
    stimuli_epoch = data.epochs['Headfix']
    analogin_start_time = stimuli_epoch.starts.times()


    """
        Now we have all the TTL epochs, we can extract the start and end of each TTL and measure the duration of the peaks and
        troughs. We then plot the peaks and the troughs in respective histograms. A perfect TTL signal should have a single 
        cluster of peaks durations around 1 second, whereas troughs can be more variable thanks to inter-trial periods.
    """ 

    troughs_start_times, troughs_end_times, troughs_durations = getEpochDuration(troughs_epochs)

    troughs_epochs_plus_durations = pd.DataFrame({'start' : np.transpose(troughs_start_times), 'end' : np.transpose(troughs_end_times), 'duration' : np.transpose(troughs_durations)})

    longest_inter_TTLs = troughs_epochs_plus_durations.loc[troughs_epochs_plus_durations['duration'] > 100]
    longest_inter_TTLs = longest_inter_TTLs.drop('duration', axis = 1)


    inspect_inter_stimuli_intervals = input("Check inter-stimuli intervals? (y/n): ")

    if inspect_inter_stimuli_intervals == 'y':
        plt.figure()
        plt.plot(ttl_tsd)
        for index, row in longest_inter_TTLs.iterrows():
            plt.axvline(x = longest_inter_TTLs.loc[index]['start'], color = 'red')
            plt.axvline(x = longest_inter_TTLs.loc[index]['end'], color = 'red')
        plt.show()

    
    inter_stim_interval_set = nap.IntervalSet(start = longest_inter_TTLs['start'], end = longest_inter_TTLs['end'], time_units = 's')

    inter_stim_interval_centers = inter_stim_interval_set.get_intervals_center()
    inter_stim_interval_centers_numpy = inter_stim_interval_centers.times()

    stimuli_starts = []
    stimuli_ends = []

    for elements in np.arange(len(inter_stim_interval_centers_numpy) - 1):
        if elements < len(inter_stim_interval_centers_numpy) - 1:
            stimuli_starts.append(inter_stim_interval_centers_numpy[elements])
            stimuli_ends.append(inter_stim_interval_centers_numpy[elements + 1])
        elif elements == len(inter_stim_interval_centers) - 1:
            stimuli_starts.append(inter_stim_interval_centers_numpy[elements])
            stimuli_ends.append(inter_stim_interval_centers_numpy[elements + 1])

    """
        Correct the start times of the intervals so the epoch start = the start of the headfix epoch analogin file. 
    """

    stimuli_interval_dataframe = pd.DataFrame({'start' : np.transpose(stimuli_starts), 'end' : np.transpose(stimuli_ends)})
    time_corrected_stimuli_dataframe = stimuli_interval_dataframe.add(analogin_start_time[0])

    stimuli_interval_set = nap.IntervalSet(start = time_corrected_stimuli_dataframe['start'], end = time_corrected_stimuli_dataframe['end'], time_units = 's')

    """
        Save stimuli epochs IntervalSet using pynapple's save function. Saving via pickle will delete important metadata from the
        intervalset and prevent pynapple from reading it. 

        I also save this as a CSV for Dom to use with his MATLAB script. The variable must be a pandas dataframe to save 
        as CSV
    """

    stimuli_interval_set.save(f'{file_location}\\{recording_basename}-Stimuli_IntervalSet')

    time_corrected_stimuli_dataframe.to_csv(f'{file_location}\\{recording_basename}-Stimuli_IntervalSet.csv')