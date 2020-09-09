import os
import numpy as np

path = "."

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith("-mtp.txt"):
            with open(os.path.join(path, filename), 'r') as txtfile:
                lines = txtfile.readlines()
                txtfile.close()
            charges = []
            for i, line in enumerate(lines):
                if i >= 4:
                    charge = float(line.split(' ')[4])
                    charges.append(charge)
            charges = np.array(charges)
            np_name = os.path.join(path, filename[:-4] + '.npy')
            np.save(np_name, charges, allow_pickle=True)
