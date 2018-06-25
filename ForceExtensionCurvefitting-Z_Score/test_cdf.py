import numpy as np
import matplotlib.pyplot as plt
import csv

F_Rup_up = np.loadtxt("C:\\Users\\brouw\\Desktop\\F_Rup_up.txt")

sorted_F_rupt = np.sort(F_Rup_up)
fraction = np.array(range(len(F_Rup_up))) / len(F_Rup_up)

fig4 = plt.figure(figsize=(10, 5))
ax8 = fig4.add_subplot(1, 1, 1)
ax8.set_xlim(0,60)
ax8.scatter(sorted_F_rupt, fraction, s=50, facecolors='none', edgecolors='r')

ax8.set_xlabel('Force (pN)')
ax8.set_ylabel('Fraction of Unfolded Nucleosomes')
ax8.set_title("Normalized Cumulative Distribution Function")

headers = ['F (pN)','Fraction']

data = np.transpose(np.concatenate((sorted_F_rupt, fraction)).reshape((2, -1)))
data = np.append(headers, data).reshape((-1, 2))

with open("C:\\Users\\brouw\\Desktop\\summary.dat", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter='\t')
    csvWriter.writerows(data)

plt.show()