#!/usr/bin/env python3
from __future__ import print_function
import os
import sys
import subprocess
import numpy as np
from mpi4py import MPI
from random import random
from lammps import lammps, PyLammps

nsamples = 50
restart = 0
name = 'ar_fcc'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ['OMP_NUM_THREADS'] = '1'

def get_dimensions(L, isprint=True, title='title'):
    lx = L.eval('lx')
    ly = L.eval('ly')
    lz = L.eval('lz')
    xy = L.eval('xy')
    yz = L.eval('yz')
    xz = L.eval('xz')
    if isprint:
        print('***** %s *****'%title)
        print('lx:', lx)
        print('ly:', ly)
        print('lz:', lz)
        print('xy:', xy)
        print('xz:', xz)
        print('ratios: %12.6f %12.6f %12.6f'%(ly/lx, lz/lx, xy/lx))
    return np.array([lx, ly, lz, xy, xz, yz])

if restart == 0:
    # initial NPT simulation to find out proper box volume
    L = PyLammps()
    L.file('%s_prep.in'%name)
    dimensions0 = None
    if rank == 0:
        dimensions0 = get_dimensions(L, title="initial")
    dimensions0 = comm.bcast(dimensions0, root=0)
    # equilibration, 10ps
    L.run(10000)
    ## take average, 50ps
    #L.run(50000)
    #L.unfix(0)
    #L.unfix(1)
    #L.undump(1)
    #if rank == 0:
    #    get_dimensions(L, title="final")
    #vols = np.array(L.runs[-1].thermo.Volume)
    #vol_current = vols[-1]
    #vol_average = np.average(vols)
    #vol_initial = L.runs[0].thermo.Volume[0]
    #if rank == 0:
    #    print('initial volume:', vol_initial)
    #    print('current volume:', vol_current)
    #    print('average volume:', vol_average)
    #scale = (vol_average/vol_initial)**0.33333333
    ## note this part of code is specific to hexaganol box
    ## rescale box to the average volume, keep hexaganol shape
    #lx, ly, lz, xy, xz, yz = dimensions0 * scale
    #L.change_box('all', 'x', 'final', 0., lx, 'y', 'final', 0., ly, 'z', 'final', 0., lz, 'xy', 'final', xy, 'xz', 'final', xz, 'yz', 'final', yz, 'remap')
    #if rank == 0:
    #    get_dimensions(L, title="scaled")
    # save restart file, preparing for real sampling
    L.write_restart('restart.%s_prep.eq'%name)
    os.system('cp restart.%s_prep.eq restart.%s.eq'%(name, name))
    del L


#################################
# Start heat conductivity calculation
#################################
# real samples, back-up files
if rank == 0:
    for folder in ['corrs', 'restarts', 'trajs', 'logs']:
        if os.path.exists(folder):
            subprocess.call(['mv', folder, folder+'.bk'])
        subprocess.call(['mkdir', folder])

for isample in range(restart, nsamples):
    L = PyLammps(cmdargs=["-screen", "none"])
    L.variable("seed1 equal %d"%int(random()*100000))
    L.variable("seed2 equal %d"%int(random()*100000))
    L.file('%s.in'%name)
    if rank == 0:
        subprocess.call(["cp", "%s.txt"%name, "logs/%s.txt.%d"%(name,isample)])
        subprocess.call(["cp", "restart.%s.eq"%name, "restarts/restart.%s.eq.%d"%(name,isample)])
        subprocess.call(["cp", "restart.%s.nve"%name, "restarts/restart.%s.nve.%d"%(name,isample)])
        subprocess.call(["cp", "nve.dcd", "trajs/nve.%d.dcd"%isample])
        subprocess.call(["cp", "corr.data", "corrs/corr.%d.data"%isample])
    del L

MPI.Finalize()
