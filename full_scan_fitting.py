import os
import numpy
import usc.analysis
import usc.models
import tkinter as tk
from tkinter import filedialog
import datetime
import matplotlib.pyplot as plt
from numpy import inf
import scipy.optimize as optimize
from typing import Tuple, Callable, Dict
from nptyping import NDArray, Float64


def model(p, f):
    Ax      = p[0]
    Ay      = p[1]
    Az      = p[2]
    offset  = p[3]
    det     = p[4]
    y       = p[5]
    Mx      = p[6]
    My      = p[7]
    Mz      = p[8]
    gx      = p[9]
    gy      = p[10]
    gz      = p[11]
    k1      = k2 = 0.5 * 193
    px      = {"d": det,  "k1": k1, "k2": k2, "y": y, "g": gx, "M": Mx, "N": 1E7}
    py      = {"d": det,  "k1": k1, "k2": k2, "y": y, "g": gy, "M": My, "N": 1E7}
    pz      = {"d": det,  "k1": k1, "k2": k2, "y": y, "g": gz, "M": Mz, "N": 1E7}
    return offset + (Ax * usc.models.Sxx(f, px)
                     + Ay * usc.models.Sxx(f, py)
                     + Az * usc.models.Sxx(f, pz))


def residuals(p: NDArray[float], x: NDArray[float], y: NDArray[float],
              f: Callable) -> float:
    return (numpy.log10(y) - numpy.log10(f(p, x)))**2 / len(x)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    load_path = "/Users/jannekhansen/Desktop/Spectra of different days /20211112 - Full_Scan/Spectra"#filedialog.askdirectory(title="Choose Spectra Location")
    save_path = "/Users/jannekhansen/Desktop/Spectra of different days /20211112 - Full_Scan/Figures"#filedialog.askdirectory(title="Choose Plot Save Location")
    files = os.listdir(load_path)
    files = [file for file in files if ".txt" in file]
    file_tree = []
    for file in files:
        pos, det, label = file.split("_")
        pos = int(pos[3:])
        det = int(det[3:-3])
        label = label.split(".")[0]
        file_tree.append((label, pos, det))
    print(file_tree)
    file_tree = sorted(file_tree, key=lambda tup: (tup[0], tup[1], tup[2]))
    print(file_tree)
    t0     = datetime.datetime.now()
    ts     = numpy.array([])
    Mxs    = []
    gxs    = []
    dets   = []
    chis   = []
    Ax     = 1.84E-3
    Ay     = 2.76E-5
    Az     = 3.9E-4
    offset = 4.8
    y      = 4
    Mx     = 193.5
    My     = 173
    Mz     = 39
    gx     = 0.21 * Mx
    gy     = 0.035 * My
    gz     = 0.23 * Mz
    files_of_interest = [file for file in file_tree
                         if ("X" in file[0]) and (file[1] == 0)]
    print(files_of_interest)
    for i, file in enumerate(files_of_interest):
        (label, pos, det) = file
        if pos != 0:
            continue
        filename = "pos{}_det{}kHz_{}.txt".format(pos, det, label)
        load_filepath = os.path.join(load_path, filename)
        f, Pxx = usc.analysis.read_data(load_filepath)
        f /= 1000
        Pxx = Pxx/Pxx[-1]
        idxs = numpy.logical_and(30 < f, f < 400)
        f_fit = f[idxs]
        Pxx_fit = Pxx[idxs]
        p_guess = [Ax, Ay, Az, offset, det, y, Mx, My, Mz, gx, gy, gz]
        chisquare_0 = sum(residuals(p_guess, f_fit, Pxx_fit, model))
        p_min = [0, 0, 0, 0, det-50, 4, Mx-5, My-5, Mz-5, 0.9*gx, 0, 0]
        p_max = [inf, inf, inf, max(Pxx), det+50, 6, Mx+5, My+5, Mz+5,
                 1.1*gx, My, Mz]
        result = optimize.least_squares(residuals, p_guess,
                                        bounds=(p_min, p_max),
                                        args=(f_fit, Pxx_fit, model))
        p = result.x
        chisquare = sum(residuals(p, f_fit, Pxx_fit, model))
        fit_guess = model(p_guess, f)
        fit = model(p, f)
        px = p
        px[1:3] = 0
        print(px[3])
        fit_x = model(px, f)
        k1 = k2 = 0.5 * 193
        px = {"d": p[4],  "k1": k1, "k2": k2, "y": p[5], "g": p[9], "M": p[6],
              "N": 1E7}
        fig, ax1 = plt.subplots(1)
        ax1.set_xlim(0, 500)
        ax1.set_ylim(1, 2E4)
        ax1.set_yscale('log', nonpositive='clip')
        ax1.plot(f, fit, "r--", alpha=0.2)
        ax1.fill_between(f, fit_x, edgecolor="r", facecolor="r", alpha=0.3) #the fit taking into account the other 2 spectra
        ax1.plot(f, fit_x, "r", alpha=1)
        ax1.plot(f[numpy.logical_not(idxs)], Pxx[numpy.logical_not(idxs)],
                 "g.", markersize=1, alpha=0.5)
        ax1.plot(f_fit, Pxx_fit, "g.", markersize=1, label=det)
        plt.title(det)
        print("{}/{}".format(i+1, len(files_of_interest)))
        print("\tDetuning: {:d}".format(det))
        print("\tGuess: {:.3g}".format(chisquare_0))
        print("\tFit: {:.3g}".format(chisquare))
        print("\n")
        plt.savefig(save_path+'/'+str(det)+".pdf",dpi=300)
        dets = numpy.append(dets, p[4])
        gxs = numpy.append(gxs, p[9]/p[6])
        chis = numpy.append(chis, chisquare)
        
    px["d"] = px["M"]
    plt.figure()
    plt.semilogy(f, usc.models.Sxx(f, px))
    plt.xlim(0, 500)
    print(numpy.mean(gxs))
    print(numpy.std(gxs))
    print(gxs)
    print(numpy.mean(chis))
    print(numpy.std(chis))
    print(chis)
    plt.show()