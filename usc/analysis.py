#!/usr/bin/env python
'''Tools for calculating the correlations in ultrastrong coupling data sets
'''

import os
import re
import numpy
from scipy import signal

__author__ = "Kahan Dare"
__credits__ = ["Kahan Dare"]
__version__ = "1.0.0"
__maintainer__ = "Kahan Dare"
__email__ = "kahan.mcaffer.dare@univie.ac.at"
__status__ = "Development"


def load_data(dname, fname, count=-1, offset=0):
    '''Loads picoscope data

    args:
        - dname (string): The relative path from this location to the data
        file.
        - fname (string): The file name of the data set.

    returns:
        - settings (dict): A dictionary containing the settings and comments
        from the header file.
        - data (numpy.array): The data in a 2D array with columns being the
        different channels
    '''
    # Reading the header file
    header_file_name = fname.replace(".bin", "_header.dat")
    try:
        settings = load_header(dname, fname=header_file_name)
    except FileNotFoundError:
        print("Could not find {}.".format(header_file_name))
        print("Attempting to load default header.")
        settings = load_header(dname)

    # Reading the data file
    data = numpy.fromfile(os.path.join(dname, fname), dtype=numpy.int16,
                          count=count, offset=offset)
    N_channels = len(settings["Channels"].keys())
    data = data.reshape((len(data)//N_channels, N_channels))
    return (settings, data)


def load_header(dname, fname="Header.dat"):
    '''Loads the header file containing all the headings

    args:
        - dname (string): The data set directory.
        - fname (string): The header file's name.

    returns:
        - settings (dict): A dictionary containing the settings and comments
        from the header file.
    '''
    settings = dict()
    channel_dict = dict()
    with open(os.path.join(dname, fname), "r") as f:
        for i, line in enumerate(f):
            try:
                label, value = re.split('=|\:', line)
                settings[label.strip()] = value.strip()
            except ValueError:
                pass
    channels = settings["Channels"]
    for channel in channels:
        label = settings.pop("Channel {}".format(channel))
        vrange = int(settings.pop("Channel {} VRange".format(channel)))
        coupling = settings.pop("Channel {} Coupling".format(channel))
        channel_dict[channel] = (label, vrange, coupling)
    settings["Channels"] = channel_dict
    settings["Resolution"] = int(settings["Resolution"])
    settings["SampleInterval"] = float(settings["SampleInterval"])
    return settings


def estimate_power_spectral_density(x,timestep, bandwidth = 1000):
    '''Takes a given timetrace and computs the power spectral desity

    args:
        - x: The given timetrace
        - timestep: The timesteps between the points
        - bandwidth: The bandwidth of the fft


    returns:
        - f (NDArray[Float64]): Array of the sample frequencies
        - Pxx (NDarray[Float64]): Array of the Power spectral density (only positive side)
    '''
    sample = 1/(timestep * bandwidth)                           #The sample size
    f, Pxx = signal.welch(x, 1 / timestep, nperseg = sample)    #uses the Signal function to calculate
    return (f, Pxx) 


