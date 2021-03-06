#!/usr/bin/env python
'''Tools for calculating the correlations in ultrastrong coupling data sets
'''

import os
import re
import numpy
import scipy.signal as signal
import scipy.stats as stats
from typing import Tuple, Callable, Dict
from nptyping import NDArray, Float64

__author__ = "Kahan Dare"
__credits__ = ["Kahan Dare"]
__version__ = "1.0.0"
__maintainer__ = "Kahan Dare"
__email__ = "kahan.mcaffer.dare@univie.ac.at"
__status__ = "Development"


def load_pico(fname: str, count=-1, offset=0) -> Tuple[dict, NDArray[float]]:
    '''Loads picoscope data

    args:
        - fname: The relative path from this location to the data file.

    returns:
        - settings: A dictionary containing the settings and comments from the
        header file.
        - data: The data in a 2D array with columns for each channel
    '''
    # Reading the header file
    header_file_name = fname.replace(".bin", "_header.dat")
    settings = load_header(header_file_name)

    # Reading the data file
    data = numpy.fromfile(fname, dtype=numpy.int16, count=count, offset=offset)
    N_channels = len(settings["Channels"].keys())
    data = data.reshape((len(data)//N_channels, N_channels))
    return (settings, data)


def load_header(fname: str) -> dict:
    '''Loads the header file containing all the headings

    args:
        - dname: The data set directory.
        - fname: The header file's name.

    returns:
        settings: A dictionary containing the settings and comments from the
        header file.
    '''
    settings = dict()
    channel_dict = dict()
    with open(fname, "r") as f:
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


def chisquare(p: NDArray[float], x: NDArray[float], y: NDArray[float],
              f: Callable) -> float:
    '''Computes the Chi-Square goodness of fit test

    args:
        - f: The fit function
        - x: The independent variable
        - y: The dependent variable
        - p: The other function parameters
    '''
    chi2, _ = stats.chisquare(f_obs=y, f_exp=f(p, x))
    return chi2


def PSD(x: NDArray[float], SI: float, bandwidth=1000, **kwargs) -> Tuple[NDArray[float], NDArray[float]]:
    '''Takes a given timetrace and computs the power spectral desity

    args:
        - x: The given timetrace
        - SI: The sample interval

    kwargs:
        - bandwidth: The bandwidth of the fft


    returns:
        - f: Array of the sample frequencies
        - Pxx: Array of the Power spectral density (only positive side)
    '''
    sample = int(1/(SI * bandwidth))  # The sample size
    kwargs["nperseg"] = sample
    f, Pxx = signal.welch(x, 1 / SI, **kwargs)
    return (f, Pxx)


def save_data(filepath: str, f: NDArray[float], Pxx: NDArray[float]):
    '''Takes the Powerspectrum and the frequency array and saves them in a txt file

    args:
        - path: where do you want the data
        - filename:
        - f: The given frequency span
        - Pxx: The given spectrum
    returns:
        - void, saves data to txt file
    '''
    numpy.savetxt(filepath, numpy.array([f, Pxx]).T, delimiter=',',
                  header="freq[Hz], pxx[Unit]")


def read_data(filepath: str):
    '''Extracts the Powerspectum and frequency from a txt file
    args:
        - path: where is the data
        - filename: how is it called
    returns:
        - f, Pxx
    '''
    f, Pxx = numpy.loadtxt(filepath, delimiter=',', usecols=(0, 1),
                           unpack=True)
    return(f, Pxx)

