#!/usr/bin/env python3
from __future__ import print_function
import sys
import numpy as np
import glob

# constants
# assume the j_x, j_y, j_z are defined as 
dt = 2.0 # fs
kB = 1.3806504e-23 # SI, m2 kg s-2 K-1
T = 200.0 # K
A2m = 1.0e-10 
fs2s = 1.0e-15
kCal2J = 4186.0/6.02214e23
volume = 26.91984*23.313265*58.49496 * (A2m)**3 # volume in SI

dest = 'corrs'

def take_last_corr(ifn):
    ifile = open(ifn, 'r')
    for line in ifile:
        words = line.split()
        if line[0] == '#':
            continue
        if len(words) == 2:
            data = []
        elif len(words) == 6:
            ipt, nlag, nsamples = [int(words[i]) for i in range(3)]
            fx, fy, fz = [float(words[i]) for i in range(3, 6)]
            data.append([nlag, fx, fy, fz])
    ifile.close()
    return data

#process = subprocess.Popen(['bash', '-i', '-c', 'ls corrs/corr.[0-9]*.data'], stdout=subprocess.PIPE)
#out, err = process.communicate()
#out = commands.getoutput('ls corrs/corr.[0-9]*.data')
words = glob.glob('%s/corr.[0-9]*.data'%dest)
words.sort()
nfiles = len(words)
iwords = []
order = int(sys.argv[1])
for i in range(order*20, (order+1)*20):
    iwords.append(words[i])
datas = []
for ifn in iwords:
    datas.append(take_last_corr(ifn))
datas = np.array(datas)

corr_average = np.average(datas, axis=0)
# convert unit to SI, heat flux: [energy]^2 * [velocity]^2
corr_average[:, 1:] *= (kCal2J * kCal2J * A2m**2 / fs2s**2) / kB / (T**2) / volume * fs2s

for d in corr_average:
    #print('%12.6f %15.8e %15.8e %15.8e'%(d[0], d[1], d[2], d[3]))
    print('%12.6f %15.8e'%(d[0]*dt, np.average(d[1:])))
