#!/usr/bin/env python3
import sys
import numpy as np
import copy

la = 5.3760
na = 6
nb = 6
nc = 6
elem = 'Ar'
#lj_cut = 13.0

masses = {
        'He': 4.0026,
        'Ne': 20.18,
        'Ar': 39.948,
        'Kr': 83.798,
        'Xe': 131.29
        }

epsilons = {
        'He': 0.02146,
        'Ne': 0.07289,
        'Ar': 0.23846,
        'Kr': 0.33981,
        'Xe': 0.43917
        }

sigmas = {
        'He': 2.570,
        'Ne': 2.790,
        'Ar': 3.380,
        'Kr': 3.600,
        'Xe': 4.100
        }

eps = 0.23810  # kcal/mol
sig = 3.405    # A
#eps = epsilons[elem]
#sig = sigmas[elem]

box_uc = np.eye(3) * la
box_sc = copy.deepcopy(box_uc)
box_sc[0] *= na
box_sc[1] *= nb
box_sc[2] *= nc

spositions_uc = np.array([
    [  0.00,  0.00,  0.00 ],
    [  0.00,  0.50,  0.50 ],
    [  0.50,  0.00,  0.50 ],
    [  0.50,  0.50,  0.00 ]
    ])

spositions_sc = []

for ia in range(na):
    for ib in range(nb):
        for ic in range(nc):
            if len(spositions_sc) == 0:
                spositions_sc = copy.deepcopy(spositions_uc)
                spositions_sc[:, 0] /= na
                spositions_sc[:, 1] /= nb
                spositions_sc[:, 2] /= nc
            else:
                spositions = copy.deepcopy(spositions_uc)
                spositions[:, 0] = spositions[:, 0]/na + ia/na
                spositions[:, 1] = spositions[:, 1]/nb + ib/nb
                spositions[:, 2] = spositions[:, 2]/nc + ic/nc
                spositions_sc = np.vstack((spositions_sc, spositions))

positions_sc = spositions_sc.dot(box_sc)

# print all atoms
print('title\n')
print('%7d atoms'%len(positions_sc))
print('      0 bonds')
print('      0 angles')
print('')
print('   1 atom types')
print('   1 bond types')
print('   1 angle types')
print('')
print('%16.9f%16.9f xlo xhi'%(0.0, la*na))
print('%16.9f%16.9f ylo yhi'%(0.0, la*nb))
print('%16.9f%16.9f zlo zhi'%(0.0, la*nc))
print('')
print('Masses\n')
print('%4d%11.6f\n'%(1, masses[elem]))
print('Pair Coeffs # lj/cut\n')
print('   1%16.9f%16.9f\n'%(eps, sig))
print('Bond Coeffs # harmonic\n')
print('   1   0.0000000000   0.0000000000 # %s-%s\n'%(elem, elem))
print('Angle Coeffs # harmonic\n')
print('   1   0.0000000000   0.0000000000 # %s-%s\n'%(elem, elem))


print('Atoms # full\n')
for ia, r in enumerate(positions_sc):
    print('%7d      1   1  0.000000%16.9f%16.9f%16.9f   0   0   0 # %s'%(ia+1, r[0], r[1], r[2], elem))
print('')


