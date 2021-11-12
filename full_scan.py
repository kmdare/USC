import matplotlib.pyplot as plt
import os
import re
import numpy
from scipy import signal
from scipy.optimize import curve_fit
import scipy.optimize 
import usc.analysis
import usc.models
import glob
from nptyping import NDArray

def residuals(coeffs: NDArray[float], y: NDArray[float], t: NDArray[float]):
    '''
    Finds the minimum difference between the fitted function and the data. 
    args: 
    coeffs: the 
    '''
    A = coeffs[0]
    p = {"y" : coeffs[1], "N" : coeffs[2], "g" : coeffs[3], "k1" : coeffs[4], "k2" : coeffs[5], "d" : coeffs[6], "M":coeffs[7]}
    return y - (A*usc.models.Sxx(t, p))

path = "/Volumes/Samsung_T5/UltraStrongData/Full_Scan/20211110 - Full_Scan/Traces/"
files = os.listdir(path)
#header = [file for file in files if ".dat" in file]
files = [file for file in files if ".bin" in file]
for file in files:
    settings,data = usc.analysis.load_pico(path+file)
    pos, det, *_ = file.split("_")
    channels = ["A","B","C","D"]
    for i in range(len(settings["Channels"])):
        f, Pxx = usc.analysis.PSD(data[:,i],settings["SampleInterval"],noverlap=0)
        usc.analysis.save_data("/Volumes/Samsung_T5/UltraStrongData/Full_Scan/20211110 - Full_Scan/Spectra/", "{}_{}_{}".format(pos,det,settings["Channels"][channels[i]][0]), f,Pxx)
        #pos = float(pos[2:])
        #det = float(det[2:-3])

'''
settings,data = usc.analysis.load_pico(tracepath+trace)
#print(np.shape(data))
x,y,z,l = data.T
f,Px = usc.analysis.PSD(x,settings["SampleInterval"])
f,Py = usc.analysis.PSD(y,settings["SampleInterval"])
f,Pz = usc.analysis.PSD(z,settings["SampleInterval"])

usc.analysis.save_data("/Volumes/Samsung_T5/UltraStrongData/Full_Scan/20211110 - Full_Scan/Spectra/", "actual_data_x_2_Hz", f,Px)
usc.analysis.save_data("/Volumes/Samsung_T5/UltraStrongData/Full_Scan/20211110 - Full_Scan/Spectra/", "actual_data_y_2_Hz", f,Py)
usc.analysis.save_data("/Volumes/Samsung_T5/UltraStrongData/Full_Scan/20211110 - Full_Scan/Spectra/", "actual_data_z_2_Hz", f,Pz)
fx, Px = usc.analysis.read_data("/Users/jannekhansen/Desktop/CalculatedData/", "actual_data_x_2_Hz")
fy, Py = usc.analysis.read_data("/Users/jannekhansen/Desktop/CalculatedData/", "actual_data_y_2_Hz")
fz, Pz = usc.analysis.read_data("/Users/jannekhansen/Desktop/CalculatedData/", "actual_data_z_2_Hz")

'''

'''
#here is the fitting done:
x0 = [10**(-6) ,1*(10**3), 10**7, 0.01, (193*10**3)/2, (193*10**3)/2, 2*10**6, 184000]
xl = [0 ,0.5*(10**3), 10**3, 0.00, (192*10**3)/2, (192*10**3)/2, 1.9*10**6, 183000]
xu = [np.inf ,20*(10**3), np.inf, 0.2, (194*10**3)/2, (194*10**3)/2, 2.1*10**6, 185000]
result = scipy.optimize .least_squares(residuals, x0 ,
                                args=( Pz[180:190],fz[180:190]), bounds=(xl, xu))
coeffs = result.x
print(result.x)
#chisquare = sum(residuals(fit_params, x=fx, y=Px))
limit=450

A = coeffs[0]   #Is the multiplication with the model the match the detection efficiency
p = {"y" : coeffs[1], "N" : coeffs[2], "g" : coeffs[3], "k1" : coeffs[4], "k2" : coeffs[5], "d" : coeffs[6], "M":coeffs[7]}
Pfitted = A* usc.models.Sxx(fx,p)

init_vals = [41727.6464, 2.0121, 1837.0656,0.0465] #this can give a hint where to look for the fit
#popt, _ = curve_fit(lorentzian, fz[0:limit], Pz[0:limit],p0=init_vals)
#popt1, _ = curve_fit(usc.models.Sxx, fz[0:limit], Pz[0:limit],p0=init_vals)
#x0,a,gam,d = popt
plt.scatter(fz[0:limit]/1000, Pz[0:limit])
#x_line = np.arange(min(fz[0:limit]), max(fz[0:limit]), 1)
#y_line = lorentzian(x_line, x0,a,gam,d)
#plt.semilogy(x_line/1000, y_line, '--', color='red')
plt.semilogy(fz[0:450]/1000, Pfitted[0:450], '--', color='red')
plt.semilogy(fz[0:450]/1000, Pz[0:limit]-Pfitted[0:450], '--', color='y')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
'''
'''
plt.semilogy(fz/1000,Pz,color="b")
plt.xlim(0,450)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.semilogy(fy/1000,Py,color="y")
plt.xlim(0,450)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

plt.semilogy(fx/1000,Px,color="r")
plt.xlim(0,450)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

plt.text(350, 10**2, "x-axis",color="r")
plt.text(350, 10, "y-axis",color="y")
plt.text(350, 1, "z-axis",color="b")
plt.show()
'''


