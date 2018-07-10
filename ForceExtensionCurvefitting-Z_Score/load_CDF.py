import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

cdf_path = r'C:\Users\tbrouwer\Desktop\CDF'

cdf_files = []
os.chdir(cdf_path)
for file in glob.glob("*.dat"):
    cdf_files.append(file)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)

colors = iter(cm.rainbow(np.linspace(0, 1, len(cdf_files))))

for file in cdf_files:
    print("Processing file: " + file)
    df = pd.read_csv(cdf_path+"\\"+file, sep='\t')
    fraction = df['Fraction']
    rupture_force = df['F (pN)']
    ax.scatter(rupture_force, fraction, edgecolors=next(colors), facecolors='none', s=50, label=str(file[:-4]))

ax.set_xlim(0,60)
ax.legend()

plt.savefig(cdf_path + "\\" + "CDF.png")
plt.show()