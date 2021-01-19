import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Bellurie per LaTeX
from matplotlib import rcParams
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preview'] = True



# Loading data file BGO1 - F18
BGO1F18 = np.loadtxt('data/dataBGO1F18.txt')
BGO2F18 = np.loadtxt('data/dataBGO2F18.txt')
LSO1Ba133 = np.loadtxt('data/dataLSO1Ba133.txt')
LSO2Ba133 = np.loadtxt('data/dataLSO2Ba133.txt')

# Plot
fig, axs = plt.subplots(2,2, figsize=(8,8))
fig.suptitle("Energy Spectrum", fontsize=14)
axs[0, 0].hist(BGO1F18,bins=200,facecolor='g',ec='black',alpha=0.5 , label='histogram data',density=False) # label='histogram data'
axs[0, 0].set_title('BGO1F18', fontsize=10)
axs[0, 0].set_xlim(0,6)
axs[0, 1].hist(BGO2F18,bins=200,facecolor='g',ec='black',alpha=0.5 , label='histogram data',density=False) # label='histogram data'
axs[0, 1].set_title('BGO1F28', fontsize=10)
axs[0, 1].set_xlim(0,5)
axs[1, 0].hist(LSO1Ba133,bins=200,facecolor='g',ec='black',alpha=0.5 , label='histogram data',density=False) # label='histogram data'
axs[1, 0].set_title('LSO1Ba133', fontsize=10)
axs[1, 0].set_xlim(1.4,3)
axs[1, 0].set_ylim(0, 5000)
axs[1, 1].hist(LSO1Ba133,bins=200,facecolor='g',ec='black',alpha=0.5 , label='histogram data',density=False) # label='histogram data'
axs[1, 1].set_title('LSO2Ba133', fontsize=10)
axs[1, 1].set_xlim(1.4,3)
axs[1, 1].set_ylim(0, 5000)

for ax in axs.flat:
    ax.set_ylabel('Counts', fontsize=10)

for ax in fig.get_axes():
    ax.label_outer()

plt.savefig('figures/multi_histograms.png', format='png',bbox_inches="tight", dpi=100)
plt.show()