from __future__ import division
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

data = np.fromfile("Traces/det50kHz_20211019_144419.bin", dtype='int16')
x,y,z,l= data.reshape((data.size//4,4)).T

start     = 47530000;
end       = 47565000;
hopp      = 47544763;
time_step = 4.8*10**(-8); 
Sample    = int(1/(time_step*1000));
############
#x[start:end]=1
part      = z[start:end];
t         = np.arange(0,np.size(part))*time_step*1000;
#plt.plot(part[::50],t[::50])
plt.plot(t,part)
plt.vlines((hopp-start)*time_step*1000, -10000,15000, colors='r')
#plt.xlim(473000,479000)
plt.xlabel("Time in s")
plt.title("Timetrace of the lost particle")
#plt.savefig("timetrace",dpi = 350)
plt.show()
##########
print(end-start);
psx = [];
arrx = [];
j=start;
i=0
freqs = np.fft.fftfreq(Sample, time_step);
idx = np.argsort(freqs);
av = 1
while j < int(hopp-Sample/2):
    psxn = np.zeros(Sample);
    psxn = psxn + (np.abs(np.fft.fft(z[int(j-Sample/2):int(Sample/2+j)]))**2)/av;
    j = j+1;
    arrx = psxn[idx]
    psx = psx + [arrx[10417:10866]];  
while j < int(hopp+(Sample/2)+1):
    psxn = np.zeros(Sample);
    psx = psx + [psxn[10417:10866]]; 
    j=j+1;
while j < end:
    psxn = np.zeros(Sample);
    psxn = psxn + (np.abs(np.fft.fft(z[int(j-Sample/2):int(Sample/2+j)]))**2)/av;
    j = j+1;
    arrx = psxn[idx]
    psx = psx + [arrx[10417:10866]];
row,col = np.array(psx).shape;
##########
levels = 10**(np.arange(7,16,0.5))
from matplotlib import ticker, cm
ya = np.arange(0, row, 1)
print(np.size(ya))
xa = freqs[idx]
#plt.xlim(0,320)
plt.xlabel("Frequency in kHz")
plt.ylabel("Time in ms")
#plt.ylim(2500,4550)
plt.contourf(xa[10417:10866]/1000, 1000*av*ya*time_step, psx, levels,locator=ticker.LogLocator())
plt.colorbar();
plt.savefig("Z_with_cut_first_hop",dpi = 300)
plt.show()