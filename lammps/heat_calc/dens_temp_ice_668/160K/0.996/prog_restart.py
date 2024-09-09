#!/usr/bin/env python3
from __future__ import print_function
import os
import sys
import subprocess
import numpy as np
from mpi4py import MPI
from random import random
from lammps import lammps, PyLammps
import glob

nsamples = 100
restart = 0
name = 'ice_Ih668'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ['OMP_NUM_THREADS'] = '1'

#def get_dimensions(L, isprint=True, title='title'):
#    lx = L.eval('lx')
#    ly = L.eval('ly')
#    lz = L.eval('lz')
#    xy = L.eval('xy')
#    yz = L.eval('yz')
#    xz = L.eval('xz')
#    if isprint:
#        print('***** %s *****'%title)
#        print('lx:', lx)
#        print('ly:', ly)
#        print('lz:', lz)
#        print('xy:', xy)
#        print('xz:', xz)
#        print('ratios: %12.6f %12.6f %12.6f'%(ly/lx, lz/lx, xy/lx))
#    return np.array([lx, ly, lz, xy, xz, yz])

#if restart == 0:
#	L = PyLammps()
#	L.file('%s_prep.in'%name)
#	dimensions0 = None
#	if rank == 0:
#		dimensions0 = get_dimensions(L, title="initial")
#	dimensions0 = comm.bcast(dimensions0, root=0)
#	L.run(10000)
#	L.write_restart('restart.%s_prep.eq'%name)
#	os.system('cp restart.%s_prep.eq restart.%s.eq'%(name, name))
#	del L
#################################
# To obtain the restart information
#################################
desk='corrs'
index = glob.glob('%s/corr.[0-9]*.data'%desk)
index.sort()
index_tar = index.pop()[11:13]
index_tar = int(index_tar)

for isample in range(index_tar + 1, nsamples):
    L = PyLammps(cmdargs=["-screen", "none"])
    L.variable("seed1 equal %d"%int(random()*100000))
    L.variable("seed2 equal %d"%int(random()*100000))
    L.variable("nname index ice_Ih668.eq.%02d"%(isample-1))
    L.file('%s_restart.in'%name)
    if rank == 0:
        subprocess.call(["cp", "%s.txt"%name, "logs/%s.txt.%02d"%(name,isample)])
        subprocess.call(["cp", "restart.%s.eq"%name, "restarts/restart.%s.eq.%02d"%(name,isample)])
        subprocess.call(["cp", "restart.%s.nve"%name, "restarts/restart.%s.nve.%02d"%(name,isample)])
        subprocess.call(["cp", "nve.dcd", "trajs/nve.%02d.dcd"%isample])
        subprocess.call(["cp", "corr.data", "corrs/corr.%02d.data"%isample])
    del L

MPI.Finalize()
