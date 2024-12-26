import os
import json
import numpy as np
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, lfilter, filtfilt

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Bandpass filtering the LFP.
    
    Parameters
    ----------
    data : Tsd/TsdFrame
        Description
    lowcut : TYPE
        Description
    highcut : TYPE
        Description
    fs : TYPE
        Description
    order : int, optional
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    time_support = data.time_support
    time_index = data.as_units('s').index.values
    if type(data) is nap.TsdFrame:
        tmp = np.zeros(data.shape)
        for i,c in enumerate(data.columns):
            tmp[:,i] = bandpass_filter(data[c], lowcut, highcut, fs, order)

        return nap.TsdFrame(
            t = time_index,
            d = tmp,
            time_support = time_support,
            time_units = 's',
            columns = data.columns)

    elif type(data) is nap.Tsd:
        flfp = _butter_bandpass_filter(data.values, lowcut, highcut, fs, order)
        return nap.Tsd(
            t=time_index,
            d=flfp,
            time_support=time_support,
            time_units='s')

    else:
        raise RuntimeError("Unknow format. Should be Tsd/TsdFrame")

def detect_oscillatory_events(lfp, epoch, freq_band, thres_band, duration_band, min_inter_duration, wsize=51):
    """
    Simple helper for detecting oscillatory events (e.g. ripples, spindles)
    
    Parameters
    ----------
    lfp : Tsd
        Should be a single channel raw lfp
    epoch : IntervalSet
        The epoch for restricting the detection
    freq_band : tuple
        The (low, high) frequency to bandpass the signal
    thres_band : tuple
        The (min, max) value for thresholding the normalized squared signal after filtering
    duration_band : tuple
        The (min, max) duration of an event in second
    min_inter_duration : float
        The minimum duration between two events otherwise they are merged (in seconds)
    wsize : int, optional
        The size of the window for digitial filtering
    
    Returns
    -------
    IntervalSet
        The intervalSet detected
    Tsd
        Timeseries containing the peaks of the oscillations
    """
    lfp = lfp.restrict(epoch)
    frequency = lfp.rate
    signal = bandpass_filter(lfp, freq_band[0], freq_band[1], frequency)
    squared_signal = np.square(signal.values)
    window = np.ones(wsize)/wsize
    nSS = filtfilt(window, 1, squared_signal)
    nSS = (nSS - np.mean(nSS))/np.std(nSS)
    nSS = nap.Tsd(t = signal.index.values, d=nSS, time_support=epoch)

    # Round1 : Detecting Oscillation Periods by thresholding normalized signal
    nSS2 = nSS.threshold(thres_band[0], method='above')
    nSS3 = nSS2.threshold(thres_band[1], method='below')

    # Round 2 : Excluding oscillation whose length < min_duration and greater than max_duration
    osc_ep = nSS3.time_support
    osc_ep = osc_ep.drop_short_intervals(duration_band[0], time_units = 's')
    osc_ep = osc_ep.drop_long_intervals(duration_band[1], time_units = 's')

    # Round 3 : Merging oscillation if inter-oscillation period is too short
    osc_ep = osc_ep.merge_close_intervals(min_inter_duration, time_units = 's')
    osc_ep = nap.IntervalSet(osc_ep.as_dataframe().reset_index(drop=True))

    # Extracting Oscillation peak
    osc_max = []
    osc_tsd = []
    for count, value in enumerate(osc_ep):
        tmp = nSS.restrict(osc_ep.loc[[count]])
        osc_tsd.append(tmp.index[np.argmax(tmp)])
        osc_max.append(np.max(tmp))

    osc_max = np.array(osc_max)
    osc_tsd = np.array(osc_tsd)

    osc_tsd = nap.Tsd(t=osc_tsd, d=osc_max, time_support=epoch)

    return osc_ep, osc_tsd

if __name__ == "__main__":

    """
        This script is a general script for detecting oscillations in the local field potential of neural recordings.

        Set up path for loading files and saving metadata.
    """    

if __name__ == '__main__':

    oscillation_type = input("What oscillation are you trying to detect? (Ripples = r, Spindles = s): ")

    if oscillation_type == 'r':
        oscillation_name = 'ripple'
        oscillation_abbreviation = 'rip'
    elif oscillation_type == 's':
        oscillation_name = 'spindle'
        oscillation_abbreviation = 'spn'        
    elif oscillation_type == '':
        raise Exception("No oscillation type entered")
    else:
        raise Exception("Entered oscillation type is not yet handled. Current valid entries are: Ripples = r, Spindles = s")

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

        print(f"Running {oscillation_name} detection for {recording_basename}")

        channel_ID_filename = f'{directory}\\{recording_basename}_{oscillation_name}_channels.json'
        
        file_exist = os.path.isfile(channel_ID_filename)

        if file_exist == True:

            print(f"{oscillation_name} channel json file found, loading target channel number.")

            open_channel_ID = open(channel_ID_filename)
            channel_IDs = json.load(open_channel_ID)

        elif file_exist == False:

            print(f"{oscillation_name} channel json file not found, using previously loaded channel ID.")

        # directory = input("Enter folder directory: ")
        # path_string = Path(directory)
        # path_parts = path_string.parts
        # recording_basename = path_parts[4]

        """
            Load the data from the specified session and extract task-relevant variables.
        """    

        data = ntm.load_session(directory, "neurosuite")

        frequency = 1250
        spikes = data.spikes
        sws_ep = data.read_neuroscope_intervals('sws')
        # sws_ep = data.epochs['Sleep']

        """
            The aim is to be flexible and allow me to detect different oscillation types with the same script.

            The following allows the user to specify which oscillations they want to detect, and tell the user whether those
            oscillation types are supported. 
        """    

        if oscillation_type == 'r':

            epoch = sws_ep
            freq_band = (100,300)
            thres_band = (7, 10)
            duration_band = (0.01,0.1)
            min_inter_duration = 0.02
            wsize=51

            test_channel = channel_IDs[0]['test_channels'][0]
            comparison_channel = channel_IDs[0]['test_channels'][0]
            
            control_channel = channel_IDs[1]['control_channels'][0]

        elif oscillation_type == 's':

            epoch = sws_ep
            freq_band = (10, 16)
            thres_band = (0.25, 20)
            duration_band = (0.4,2.1)
            min_inter_duration = 0.02
            wsize=51

            test_channel = channel_IDs[0]['test_channels'][0]
            comparison_channel = channel_IDs[0]['test_channels'][2]
            
            control_channel = channel_IDs[1]['control_channels'][0]

        """
           Load LFP for this folder.    
        """    

        lfp = data.load_lfp(channel=test_channel,extension='.eeg',frequency=frequency)
        lfp2 = data.load_lfp(channel=comparison_channel,extension='.eeg',frequency=frequency)        
        control_lfp = data.load_lfp(channel=control_channel,extension='.eeg',frequency=frequency)

        # noise_thres_band = (1, 7)

        # noise_ep, noise_tsd = detect_oscillatory_events(control_lfp, sws_ep, freq_band, noise_thres_band, duration_band, min_inter_duration, wsize)

        # denoised_lfp = lfp.set_diff(noise_ep)

        """
            Detect oscillatory events with parameters set for the prespecified event types    
        """    

        oscillation_ep, oscillation_tsd = detect_oscillatory_events(lfp, sws_ep, freq_band, thres_band, duration_band, min_inter_duration, wsize)

        print(f"found {len(oscillation_ep)} {oscillation_name}s")

        # ex_ep = nap.IntervalSet(start = 1027.00, end = 1029.0, time_units = 's') 
        # lfpsleep = lfp.restrict(sws_ep)
        # # plt.figure(figsize=(15,5))
        # # plt.plot(lfpsleep.restrict(ex_ep).as_units('s'))
        # # plt.xlabel("Time (s)")
        # # plt.show()

        # signal = bandpass_filter(lfpsleep, 10, 16, frequency)

        # # plt.figure(figsize=(15,5))
        # # plt.subplot(211)
        # # plt.plot(lfpsleep.restrict(ex_ep).as_units('s'))
        # # plt.subplot(212)
        # # plt.plot(signal.restrict(ex_ep).as_units('s'))
        # # plt.xlabel("Time (s)")
        # # plt.show()

        # windowLength = 51
        # squared_signal = np.square(signal.values)
        # window = np.ones(windowLength)/windowLength
        # nSS = filtfilt(window, 1, squared_signal)
        # nSS = (nSS - np.mean(nSS))/np.std(nSS)
        # nSS = nap.Tsd(t=signal.index.values, 
        #               d=nSS, 
        #               time_support=signal.time_support)
                      

        # low_thres = 0.25
        # high_thres = 10

        # nSS2 = nSS.threshold(low_thres, method='above')
        # nSS3 = nSS2.threshold(high_thres, method='below')

        # plt.figure(figsize=(15,5))
        # plt.subplot(311)
        # plt.plot(lfpsleep.restrict(ex_ep).as_units('s'))
        # plt.subplot(312)
        # plt.plot(signal.restrict(ex_ep).as_units('s'))
        # plt.subplot(313)
        # plt.plot(nSS.restrict(ex_ep).as_units('s'))
        # plt.plot(nSS.restrict(ex_ep).as_units('s'))
        # plt.plot(nSS3.restrict(ex_ep).as_units('s'), '.')
        # plt.axhline(low_thres)
        # plt.xlabel("Time (s)")
        # plt.tight_layout()
        # plt.show()

        # Save in .evt file for Neuroscope
        start = oscillation_ep.as_units('ms')['start'].values
        peaks = oscillation_tsd.as_units('ms').index.values
        ends = oscillation_ep.as_units('ms')['end'].values

        datatowrite = np.vstack((start,peaks,ends)).T.flatten()

        n = len(oscillation_ep)

        texttowrite = np.vstack(((np.repeat(np.array([f'{oscillation_abbreviation} start 1']), n)),
                                (np.repeat(np.array([f'{oscillation_abbreviation} peak 1']), n)),
                                (np.repeat(np.array([f'{oscillation_abbreviation} stop 1']), n))
                                    )).T.flatten()

        evt_file = os.path.join(path_string, data.basename + f'.evt.py.{oscillation_abbreviation}')
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()

        oscillation_tsd.save(os.path.join(path_string, data.basename + f'_{oscillation_name}_tsd'))
        oscillation_ep.save(os.path.join(path_string, data.basename + f'_{oscillation_name}_ep'))