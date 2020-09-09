import os
import subprocess

count = 1

for filename in sorted(os.listdir(os.getcwd())):
    if filename.endswith('.in'):
        if count >= 1 and count <= 500:
            subprocess.call(['submit4', 'p10', filename])

        count += 1

print(len(os.listdir(os.getcwd())))
