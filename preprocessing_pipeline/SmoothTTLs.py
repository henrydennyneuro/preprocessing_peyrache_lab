import os
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt

from pathlib import Path
from ValidateTTLs import extractTTL, getEpochDuration

if __name__ == '__main__':

    """
        This script is designed to smooth out periods where TTL detection was interrupted and save the IntervalSet as a pcikle
        file. 

        For TTL extraction, you must give the exact analogin filename where the TTL was recorded. Giving the folder only will
        not work, and will crash the script.

        First we will extract all the filenaming parameters we need to save the IntervalSet to pickle
    """    

    analogin_file = input("Enter analogin dat file location: ")
    
    file_location = os.path.dirname(analogin_file)

    path_string = Path(analogin_file)
    path_parts = path_string.parts
    recording_basename = path_parts[4]

    """
        Extract TTL. 
    """    

    ttl = extractTTL(analogin_file)
    ttl_tsd = nap.Tsd(ttl)


    """
        Now detect the TTL peaks and troughs and create intervalsets for each. 
    """    

    ttl_threshold = 30000
    detected_peaks = ttl_tsd.threshold(ttl_threshold)
    peaks_epochs = detected_peaks.time_support
    troughs_epochs = ttl_tsd.time_support.set_diff(peaks_epochs)

    """
        Smooth TTL's by merging close peaks less than 1 TTL interval. 
    """    

    smoothed_peaks_epochs = peaks_epochs.merge_close_intervals(0.2)
    smoothed_troughs_epochs = ttl_tsd.time_support.set_diff(smoothed_peaks_epochs)
    troughs_minus_interstim_epochs = smoothed_troughs_epochs.drop_long_intervals(100) # Drop the inter-stimuli intervals
    # as they compress the histogram and make it useless. 

    """
        Retrieve TTL time perameters and plot as a histigram to check and see if distribution of intervals seem correct. 
    """

    peaks_start_times_post_smooth, peaks_end_times_post_smooth, peaks_durations_post_smooth = getEpochDuration(smoothed_peaks_epochs)
    troughs_start_times_post_smooth, troughs_end_times_post_smooth, troughs_durations_post_smooth = getEpochDuration(troughs_minus_interstim_epochs)

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.hist(peaks_durations_post_smooth, bins = 100)
    ax2.hist(troughs_durations_post_smooth, bins = 100)
    ax1.title.set_text("Distribution of TTL Peak Durations")
    ax2.title.set_text("Distribution of TTL Trough Durations")
    plt.show()

    peaks_centers = smoothed_peaks_epochs.get_intervals_center()
    troughs_centers = troughs_minus_interstim_epochs.get_intervals_center()

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(peaks_centers.times(), peaks_durations_post_smooth)
    ax2.scatter(troughs_centers.times(), troughs_durations_post_smooth)
    ax1.title.set_text("Distribution of TTL Peak Durations")
    ax2.title.set_text("Distribution of TTL Trough Durations")
    plt.show()

    """
        Create a Pandas Dataframe from the time parameters so we can plot the start and end of each TTL on top of the raw TTL. 
    """   

    peaks_epochs_plus_durations_post_smooth = pd.DataFrame({'start' : np.transpose(peaks_start_times_post_smooth), 'end' : np.transpose(peaks_end_times_post_smooth), 'duration' : np.transpose(peaks_durations_post_smooth)})
    troughs_epochs_plus_durations_post_smooth = pd.DataFrame({'start' : np.transpose(troughs_start_times_post_smooth), 'end' : np.transpose(troughs_end_times_post_smooth), 'duration' : np.transpose(troughs_durations_post_smooth)})

    TTLs_post_smooth = peaks_epochs_plus_durations_post_smooth.drop('duration', axis = 1)

    longest_TTLs_post_smooth = peaks_epochs_plus_durations_post_smooth.loc[peaks_epochs_plus_durations_post_smooth['duration'] > 0.51]
    longest_TTLs_post_smooth = longest_TTLs_post_smooth.drop('duration', axis = 1)
    
    shortest_TTLs_post_smooth = peaks_epochs_plus_durations_post_smooth.loc[peaks_epochs_plus_durations_post_smooth['duration'] < 0.375]
    shortest_TTLs_post_smooth = shortest_TTLs_post_smooth.drop('duration', axis = 1)

    """
        This first plot will tell you if you have any unusual TTLs (red = Unexpectedly long TTLs, green = suspected interrupted 
        TTLs)
    """

    # plt.figure()
    # plt.plot(ttl_tsd)
    # for index, row in longest_TTLs_post_smooth.iterrows():
    #     plt.axvline(x = longest_TTLs_post_smooth.loc[index]['start'], color = 'red')
    #     plt.axvline(x = longest_TTLs_post_smooth.loc[index]['end'], color = 'red')
    # for index, row in shortest_TTLs_post_smooth.iterrows():
    #     plt.axvline(x = shortest_TTLs_post_smooth.loc[index]['start'], color = 'green')
    #     plt.axvline(x = shortest_TTLs_post_smooth.loc[index]['end'], color = 'green')
    # plt.show()

    # """
    #    Plot start (green) and end (red) of every TTL. Check to see if they look appropriate. 
    # """

    # plt.figure()
    # plt.plot(ttl_tsd)
    # for index, row in TTLs_post_smooth.iterrows():
    #     plt.axvline(x = TTLs_post_smooth.loc[index]['start'], color = 'green')
    #     plt.axvline(x = TTLs_post_smooth.loc[index]['end'], color = 'red')
    # plt.show()

    """
        TTLs currently start from t=0, however the visual stimuli is the second recording epoch. When the recording DAT files
        are concatenated by the spike sorter, the second recording epoch has a new start time. Because the spike sorter does
        not concatinate the analogin TTL files, the analogin file's start time needs to be corrected. In order to align the TTLs
        with the recording DAT file, we must find the start of the visual stim epoch, and add that value to all the analogin
        times. This will align the TTL with the stims. 
    """

    data = ntm.load_session(file_location, "neurosuite")

    stimuli_epoch = data.epochs['Headfix']
    analogin_start_time = stimuli_epoch.starts.times()

    TTLs_start_times = nap.IntervalSet(TTLs_post_smooth).starts.times(units = 's')
    TTLs_end_times = nap.IntervalSet(TTLs_post_smooth).ends.times(units = 's')

    corrected_TTLs_start_times = TTLs_start_times + analogin_start_time
    corrected_TTLs_end_times = TTLs_end_times + analogin_start_time
    
    corrected_TTLs_dataframe = pd.DataFrame({'start' : np.transpose(corrected_TTLs_start_times), 'end' : np.transpose(corrected_TTLs_end_times)})
    corrected_TTLs_interval_set = nap.IntervalSet(start = corrected_TTLs_dataframe['start'], end = corrected_TTLs_dataframe['end'], time_units = 's')

    """
        Save TTL IntervalSet using pynapple's save function. Saving via pickle will delete important metadata from the
        intervalset and prevent pynapple from reading it. 
    """

    corrected_TTLs_interval_set.save(f'{file_location}\\{recording_basename}-TTL_IntervalSet')

    """
        Convert smoothed TTLs to pandas dataframe and save TTL IntervalSet dataframe as CSV. 
    """    

    corrected_TTLs_dataframe.to_csv(f'{file_location}\\{recording_basename}-TTL_IntervalSet.csv')