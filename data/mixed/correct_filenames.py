import glob, os, sys

#for filename in glob.glob('*.xyz'):
#    if 'dsgdb9nsd' in filename:
#        identifier = filename.split('.')[0].split('_')[1]
#        if not os.path.exists(filename[:-4] + '.npy'):
#            os.rename(f'{identifier}_0_mbis-mtp.npy', filename[:-4] + '.npy')
#    elif 'psi4-opt-b3lyp' in filename:
#        name_abbrev = filename.split('.xy')[0]
#        if not os.path.exists(f'{filename[:-4]}.npy'):
#            os.rename(f'{name_abbrev[:-2]}_mbis-mtp.npy', f'{filename[:-4]}.npy')
#    elif 'SSI-' in filename:
#        name = filename[:-4]
#        if not os.path.exists(f'{name}.npy'):
#            os.rename(f'{name}_mbis-mtp.npy', f'{name}.npy')
#    elif 'psi4-opt-b3lyp' in filename and filename.startswith('y'):
#        name_abbrev = filename.split('.x')[0]
#        if not os.path.exists(f'{filename[:-4]}.npy'):
#            os.rename(f'{name_abbrev[1:-2]}_mbis-mtp.npy', f'{filename[:-4]}.npy')


for filename in glob.glob('*.xyz'):
    if 'dsgdb9nsd' in filename:
        with open('temp.xyz', 'w+') as tempfile:
            with open(filename, 'r') as readfile:
                xyzlines = readfile.readlines()
                readfile.close()
            file_len = len(xyzlines)
            for i, line in enumerate(xyzlines):
                if i==0:
                    tempfile.write(line)
                elif i==1:
                    tempfile.write("0 0\n")
                elif i < file_len-3:
                    tempfile.write(line)
            tempfile.close()
        os.rename('temp.xyz', filename)

    elif 'psi4-opt-b3lyp' in filename:
        negative=False
        positive=False
        if os.path.isfile(f'../data/opt_chargedn/{filename}'):
            negative=True
        elif os.path.isfile(f'../data/opt_chargedp/{filename}'):
            positive=True

        with open('temp.xyz', 'w+') as tempfile:
            with open(filename, 'r') as readfile:
                xyzlines = readfile.readlines()
                readfile.close()
            file_len = len(xyzlines)
            for i, line in enumerate(xyzlines):
                if i==0:
                    tempfile.write(line)
                elif i==1 and positive:
                    tempfile.write("1 0\n")
                elif i==1 and negative:
                    tempfile.write("-1 0\n")
                else:
                    tempfile.write(line)
            tempfile.close()
        os.rename('temp.xyz', filename)

