import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt

def extractTTL(file, n_channels = 1, channel = 0, fs = 20000):
    """
        load ttl from analogin.dat
    """
    n_channels = 1

    f = open(file, 'rb')
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2        
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    sample_rate = 20000
    f.close()
    
    with open(file, 'rb') as f:
        data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
    if n_channels == 1:
        data = data.flatten().astype(np.int32)
    else:
        data = data[:,channel].flatten().astype(np.int32)
    
    ttl = pd.Series(data) 

    ttl_index_values = list(ttl.index.values)
    ttl_index_values_rescaled = np.array(ttl_index_values)/sample_rate

    ttl_data = ttl.tolist()

    ttl_time_rescaled = pd.Series(data = ttl_data, index = ttl_index_values_rescaled) 

    return ttl_time_rescaled

    ## Run

def getEpochDuration(epoch):
    """
        Calculate the the duration of every epoch in a given pynapple interval set
    """    
    epoch = epoch
    
    epoch_starts = epoch.starts
    epoch_ends = epoch.ends

    epoch_start_times = epoch_starts.times(units = 's')
    epoch_end_times = epoch_ends.times(units = 's')

    epochs_durations = np.subtract(epoch_end_times, epoch_start_times)

    return epoch_start_times, epoch_end_times, epochs_durations


if __name__ == '__main__':

    """
        The aim of this script is to check the parameters of TTL's recorded during the presentation of visual landmarks and
        their DIFFEO scrambled counterparts to head-fixed mice. To align the images with the neural data, we attached a
        photodiode to the presentation screen and used it to drive an arduino which sent a TTL pulse to Intan. Reliability
        of this method is variable, as parameters such as photodiode sensitivity and ambient light can affect the photodiode
        output. This script looks for conjoined TTL's, dropped TTL's, and interrupted TTL's and notifies the user if any are 
        detected. 

        For TTL extraction, you must give the exact analogin filename where the TTL was recorded. Giving the folder only will
        not work, and will crash the script. 
    """    

    location = input("Enter analogin dat file location: ")
    
    """
        Extract TTL and convert it into a pynapple timeseries to make it easier to work with. We will use the pynapple threshold
        function to find the TTL voltage peaks and the time_support method to find consecutive supra-threshold values. We then 
        record the start/end timestamps of both the troughs and the peaks. This will allow us to check TTL's are stable and that
        no TTL's are dropped.
    """    

    ttl = extractTTL(location)

    ttl_tsd = nap.Tsd(ttl)

    ttl_threshold = 30000
    detected_peaks = ttl_tsd.threshold(ttl_threshold)
    peaks_epochs = detected_peaks.time_support
    troughs_epochs = ttl_tsd.time_support.set_diff(peaks_epochs)

    """
        Now we have all the TTL epochs, we can extract the start and end of each TTL and measure the duration of the peaks and
        troughs. We then plot the peaks and the troughs in respective histograms. A perfect TTL signal should have a single 
        cluster of peaks durations around 1 second, whereas troughs can be more variable thanks to inter-trial periods.
    """ 

    peaks_start_times, peaks_end_times, peaks_durations = getEpochDuration(peaks_epochs)
    troughs_start_times, troughs_end_times, troughs_durations = getEpochDuration(troughs_epochs)


    print("Plotting the distribution of TTL peaks and troughs.")

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.hist(peaks_durations, bins = 100)
    ax2.hist(troughs_durations, bins = 100)
    ax1.title.set_text("Distribution of TTL Peak Durations")
    ax2.title.set_text("Distribution of TTL Trough Durations")
    plt.show()

    """
        If there are any unusual TTL's, it's important to visually examine them. We will first compile all the TTL time data
        into one dataframe. Then, by ordering the TTL peaks by length in descending order, we can find any unusually long TTLs 
        where the photodiode may have remained above threshold for too long. We will then restrict the raw TTL to that epoch 
        (+- 30 seconds) and plot it for visual examination.
    """ 

    peaks_epochs_plus_durations = pd.DataFrame({'start' : np.transpose(peaks_start_times), 'end' : np.transpose(peaks_end_times), 'duration' : np.transpose(peaks_durations)})
    troughs_epochs_plus_durations = pd.DataFrame({'start' : np.transpose(troughs_start_times), 'end' : np.transpose(troughs_end_times), 'duration' : np.transpose(troughs_durations)})

    longest_TTLs = peaks_epochs_plus_durations.loc[peaks_epochs_plus_durations['duration'] > 15000]
    longest_TTLs = longest_TTLs.drop('duration', axis = 1)
    
    shortest_TTLs = peaks_epochs_plus_durations.loc[peaks_epochs_plus_durations['duration'] < 0.1]
    shortest_TTLs = shortest_TTLs.drop('duration', axis = 1)

    recommend_inspect_TTLs = 'n'

    if len(longest_TTLs) > 0:
        print(str(len(longest_TTLs))+" conjoined TTL's found.")
        recommend_inspect_TTLs = 'y'
    else:
        print("No conjoined TTL's found.")

    if len(shortest_TTLs) > 0:
        print(str(len(shortest_TTLs))+" TTL interruptions found.")
        recommend_inspect_TTLs = 'y'
    else:
        print("No TTL interruptions found.")

    if recommend_inspect_TTLs == 'y':
        inspect_TTLs = input("We recommend you visually inspect your TTL's. Plot TTL trace and highlight unusual TTL's? (y/n):")

    """
        Plot the longest TTL's with an expanded window to include the edges of the TTL. I deprecated this analysis, as simply 
        plotting the start/ends of peaks and troughs was more convenient and equivalent. 
    """ 
    # window = 100000
    # count = 0
    # plt.figure()
    # for index, row in longest_TTLs.iterrows():
    #     plt.subplot(3, 10, count+1)
    #     expand_epoch = nap.IntervalSet(start = longest_TTLs.loc[index]['start'] + window, end = longest_TTLs.loc[index]['end'] + window)
    #     ttl_segment = ttl_tsd.restrict(peaks_epochs.iloc[[index]])
    #     plt.plot(ttl_segment)
    #     count = count + 1
    # plt.show()

    """
        Plot the boundaries of the longest and shortest TTL's on top of the raw TTL
    """
    if inspect_TTLs == 'y':
        plt.figure()
        plt.plot(ttl_tsd)
        for index, row in longest_TTLs.iterrows():
            plt.axvline(x = longest_TTLs.loc[index]['start'], color = 'red')
            plt.axvline(x = longest_TTLs.loc[index]['end'], color = 'red')
        for index, row in shortest_TTLs.iterrows():
            plt.axvline(x = shortest_TTLs.loc[index]['start'], color = 'green')
            plt.axvline(x = shortest_TTLs.loc[index]['end'], color = 'green')
        plt.show()

    """
        Now we've checked the TTL lengths, we can get the inter-stimuli epochs and look for skipped TTL's (i.e, periods where images were displayed but not
        detected by the photodiode).
    """

    longest_inter_TTLs = troughs_epochs_plus_durations.loc[troughs_epochs_plus_durations['duration'] > 200001]
    longest_inter_TTLs = longest_inter_TTLs.drop('duration', axis = 1)

    print("Found " + str(len(longest_inter_TTLs)) + " inter-stimuli intervals")

    inspect_inter_stimuli_intervals = input("Check inter-stimuli intervals? (y/n): ")

    if inspect_inter_stimuli_intervals == 'y':
        plt.figure()
        plt.plot(ttl_tsd)
        for index, row in longest_inter_TTLs.iterrows():
            plt.axvline(x = longest_inter_TTLs.loc[index]['start'], color = 'red')
            plt.axvline(x = longest_inter_TTLs.loc[index]['end'], color = 'red')
        plt.show()

    dropped_TTLs = troughs_epochs_plus_durations.loc[(troughs_epochs_plus_durations['duration'] > 55000) & (troughs_epochs_plus_durations['duration'] < 200001)]
    
    recommend_inspect_dropped_TTLs = 'n'
    inspect_dropped_TTLs = 'n'

    if len(dropped_TTLs) > 0:
        print("Found "+str(len(dropped_TTLs))+" potential dropped TTL's")
        recommend_inspect_dropped_TTLs = 'y'
    else:
        print("No dropped TTL's found.")

    if recommend_inspect_dropped_TTLs == 'y':
        inspect_dropped_TTLs = input("We recommend you visually inspect for dropped TTL's. Plot TTL trace and highlight unusual gaps? (y/n):")

    if inspect_dropped_TTLs == 'y':
        plt.figure()
        plt.plot(ttl_tsd)
        for index, row in dropped_TTLs.iterrows():
            plt.axvline(x = dropped_TTLs.loc[index]['start'], color = 'green')
            plt.axvline(x = dropped_TTLs.loc[index]['end'], color = 'green')
        plt.show()
