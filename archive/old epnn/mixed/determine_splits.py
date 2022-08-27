import numpy as np
import scipy.spatial
import os

for filename in os.listdir('./'):
    if os.path.exists(filename[:-4]+'.npy') and filename.endswith('.xyz') and filename.startswith('SSI'):
        xyzfile = open(filename, 'r')
        lines = xyzfile.readlines()
        xyz = []
        for line in lines[2:]:
            data = line.split()
            elem_name = data[0]
            xyz.append([data[1], data[2], data[3]])
        xyz = np.array(xyz, dtype=np.float32)
        max_distance = 0.0
        for i in range(len(xyz)):
            if i>1 and i<len(xyz):
                monA = xyz[:i]
                monB = xyz[i:]
                DA = scipy.spatial.distance_matrix(monA, monA)
                DB = scipy.spatial.distance_matrix(monB, monB)
                DAB = scipy.spatial.distance_matrix(monA, monB)
                #intramon_dists = np.mean(DA) + np.mean(DB)
                #if intramon_dists < min_distance:
                #    min_distance = intramon_dists
                #    split_ind = i
                intermon_dists = np.max(DAB)
                if intermon_dists > max_distance:
                    max_distance = intermon_dists
                    split_ind = i
        np.save(filename[:-4] + 'splits.npy', split_ind)
