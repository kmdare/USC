import os
import numpy
import usc.analysis
import usc.models
import tkinter as tk
from tkinter import filedialog
import datetime


def PSD(filepath, bandwidth=1000, N_chunk_avgs=1):
    header_file_name = filepath.replace(".bin", "_header.dat")
    settings = usc.analysis.load_header(header_file_name)
    SI = settings["SampleInterval"]
    N_chs = len(settings["Channels"])
    N_sample = int(1/(SI * bandwidth))  # Number of samples for the bandwidth
    N_chunk = N_sample * N_chunk_avgs  # Number of samples to read at once
    # Number of samples to read at once for all channels
    count = N_chs * N_chunk
    file_size = os.path.getsize(filepath)  # in bytes
    N_total = file_size / 16. / len(settings["Channels"])  # Number of samples
    N_avgs = N_total / N_sample
    N_done = 0  # in terms of samples for a channel
    Pii = None
    while N_done/N_sample < N_avgs:
        offset = 2 * N_chs * N_done  # In units of bytes
        _, data = usc.analysis.load_pico(filepath, count=count, offset=offset)
        for i in range(data.shape[1]):
            f, Pxx = usc.analysis.PSD(data[:, i], SI, bandwidth=bandwidth,
                                      noverlap=int(0.9*N_sample))
            if Pii is None:
                Pii = numpy.array(data.shape[1]*[0])
                Pii = numpy.zeros((len(Pxx), data.shape[1]))
            Pii[:, i] = (Pii[:, i]+Pxx)/N_avgs
        N_done += N_chunk
    return settings, f, Pii


if __name__ == "__main__":
    CHANNELS = ["A", "B", "C", "D"]
    OVERWRITE = True

    root = tk.Tk()
    root.withdraw()

    load_path = filedialog.askdirectory(title="Choose Data Location")
    save_path = filedialog.askdirectory(title="Choose Spectra Save Location")
    files = os.listdir(load_path)
    files = [file for file in files if ".bin" in file]
    t0 = datetime.datetime.now()
    ts = numpy.array([])
    for i, file in enumerate(files):
        load_filepath = os.path.join(load_path, file)
        pos, det, *_ = file.split("_")

        # settings, data = usc.analysis.load_pico(data_path)
        # SI = settings["SampleInterval"]
        # for j in range(len(settings["Channels"])):
        #     ch_label = settings["Channels"][CHANNELS[j]][0]
        #     save_file = "{}_{}_{}.txt".format(pos, det, ch_label)
        #     save_filepath = os.path.join(save_path, save_file)
        #     if os.path.isfile(save_filepath) and not OVERWRITE:
        #         print("\tSkipped: {}".format(save_file))
        #     else:
        #         f, Pxx = usc.analysis.PSD(data[:, j], SI, noverlap=0)
        #         usc.analysis.save_data(save_filepath, f, Pxx)
        settings, f, Pii = PSD(load_filepath,bandwidth=500, N_chunk_avgs=200)
        for j in range(Pii.shape[1]):
            ch_label = settings["Channels"][CHANNELS[j]][0]
            save_file = "{}_{}_{}.txt".format(pos, det, ch_label)
            save_filepath = os.path.join(save_path, save_file)
            usc.analysis.save_data(save_filepath, f, Pii[:, j])
            if j > 4:
                print("PROBLEM")
                raise ValueError
        try:
            t1 = datetime.datetime.now()
            dt = (t1 - t0).total_seconds()
            ts = numpy.append(ts, dt)
            t0 = t1
            N = len(files)-i-1
            weights = numpy.exp(-2 * numpy.arange(len(ts)-1, -1, -1)/N)
            dt_mean = numpy.average(ts, weights=weights)
            time_remaining = datetime.timedelta(seconds=dt_mean*N)
            print("{} / {} : Time remaining: {}".format(i+1, len(files),
                                                        str(time_remaining)))
        except ValueError:
            pass