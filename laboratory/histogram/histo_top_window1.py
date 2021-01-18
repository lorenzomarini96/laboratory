import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Bellurie per LaTeX
from matplotlib import rcParams
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preview'] = True

from scipy.optimize import curve_fit
from scipy import stats

# Loading data file BGO1 - F18
BGO1F18 = np.loadtxt('data/dataBGO1F18.txt')

# NEW PLOT 
fig = plt.figure(1, figsize = (10,6)) 
ax1 = fig.add_axes([0,0,1,1])
# Entries, Media, Standard deviation, Errore sulla media
entries    = len(BGO1F18)
mean, std  = np.mean(BGO1F18), np.std(BGO1F18)
mean_error = std/np.sqrt(len(BGO1F18))
# Creazione dell'istogramma (con barre di errore)
bin_heights, bin_borders, _ = plt.hist(BGO1F18,bins=200,facecolor='g',ec='black',alpha=0.5 , label='histogram data',density=False) # label='histogram data'
bin_centers                 = 1/2 * (bin_borders[1:] + bin_borders[:-1])
sigma_heights               = np.sqrt(bin_heights) # Errore Poissoniano
ax1.errorbar(bin_centers, bin_heights, sigma_heights, fmt='.', color='black', ecolor='black') # label='error'

ax1.set_xlim(-0.5,10)
ax1.set_ylim(0,5000)
ax1.set_xlabel('adc')
ax1.set_ylabel('counts')
ax1.set_title('BGO1F18')
ax1.minorticks_on()
ax1.legend(frameon=False ,fancybox=True, shadow=False, loc='best', prop={"size":12}, numpoints = 1)

######################################################
ax2 = fig.add_axes([0.5,0.5,0.5, 0.5])
bin_heights, bin_borders, _ = plt.hist(BGO1F18,bins=200,facecolor='g',ec='black',alpha=0.5 ,density=False) # label='histogram data'
bin_centers                 = 1/2 * (bin_borders[1:] + bin_borders[:-1])
sigma_heights               = np.sqrt(bin_heights) # Errore Poissoniano
plt.errorbar(bin_centers, bin_heights, sigma_heights, fmt='.', color='black', ecolor='black') # label='error'
# FIT GAUSSIANO
# Definisco gli estremi di intervallo su cui eseguire il fit gaussiano
a = 2.9
b = 4.7
# Creazione dei vettori per il fit sul fotopicco => elimino la parte che non interessa
bin_centers_new = []
bin_heights_new = []

for i in range(len(bin_heights)):
    if bin_centers[i] < b and bin_centers[i] > a:
        bin_centers_new.append(bin_centers[i])
        bin_heights_new.append(bin_heights[i])
    else:
        pass

bin_centers_new   = np.array(bin_centers_new)
bin_heights_new   = np.array(bin_heights_new)
sigma_heights_new = np.sqrt(bin_heights_new)

#================================================
# Funzione di fit
#================================================
def fit_gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
    
# Parametri iniziali
param0     = [entries, mean, std]
# Best Parameters 
popt, pcov = curve_fit(fit_gauss, bin_centers_new, bin_heights_new, param0, sigma = sigma_heights_new)

# Best-Parameters
A           = popt[0]
sigma_A     = np.sqrt(pcov[0,0])
mu          = popt[1]
sigma_mu    = np.sqrt(pcov[1,1])
sigma       = popt[2]
sigma_sigma = np.sqrt(pcov[2,2])

# Definizione vettore delle x (asse x di estremi minimo e massimo di x_data)
x = np.linspace(min(bin_centers),max(bin_centers),500)
#x = np.linspace(min(bin_centers_new),max(bin_centers_new),100)

# Plot fit 
ax2.plot(x, fit_gauss(x, *popt), '-', color="blue", label='Gauss')


#================================================
# CHI2 TEST
#================================================
chi2    = sum(((bin_heights_new - fit_gauss(bin_centers_new, *popt)) / sigma_heights_new)**2)
# Numero di gradi di libertà
dof     = len(bin_centers_new) - len(param0) - 1 # Sottraggo 1 perché N costante
# Calcolo dei chi2 ridotto
chi2_rid = chi2/dof
# Calcolo del p-value
pvalue = 1 - stats.chi2.cdf(chi2, dof) # pvalue deve essere maggiore di 0.05

# Full Width at Half Maximum
FWHM       = 2*np.sqrt(2*np.log(2))*sigma
sigma_FWHM = 2*np.sqrt(2*np.log(2))*sigma_sigma

# Risuoluzione 
R       = (FWHM/mu) * 100      # %Energy resolution = FWHM x 100 /photo peak
sigma_R = R * (np.sqrt((sigma_FWHM/FWHM)**2 + (sigma_mu/mu)**2))  # error propagation
 

# Bellurie 
ax2.set_xlabel('adc')
ax2.set_ylabel('counts')

ax2.set_xlim(2.5,4.7) 
ax2.minorticks_on()
ax2.set_title('BGO1F18 - Gauss Fit')
#ax2.set_ylim(0,5000)


ax2.plot([], [], color='white', marker='.',linestyle='None', label=r'$\chi^2$/$dof$  %.2f/%.i'      %(chi2,dof))
ax2.plot([], [], color='white', marker='.',linestyle='None', label=r'$\mu$    (%.2f $\pm$ %.2f)'          %(mu, sigma_mu))
ax2.plot([], [], color='white', marker='.',linestyle='None', label=r'$FWHM$  (%.2f $\pm$ %.2f)'    %(FWHM, sigma_FWHM))
#plt.plot([], [], color='white', marker='.',linestyle='None', label=r'$\delta$ E/E     (%.2f $\pm$ %.2f) %%'       %(R, sigma_R))
ax2.legend(frameon=False ,fancybox=True, shadow=False, loc='best', prop={"size":12}, numpoints = 1)

plt.savefig('figures/BGO1F18_gauss_new.png', format='png',bbox_inches="tight", dpi=100) 
#plt.show()


#================================================
# STAMPA RISULTATI DEL FIT 
#================================================
print('=================================================\n')
print('Entries   %.i'   %entries)
print('Mean      %.3f'  %mean)
print('Std Dev   %.3f'  %std)
print('\n#--------------------------------\n')
print('# Fit Gaussiano:')
print('p(x) = A * exp(-(x-mu)^2/(2 * sigma^2)\n')
print('# Parametri del fit:')
print('A        (%.3f ± %.3f)' %(popt[0],np.sqrt(pcov[0,0])))
print('mu       (%.3f ± %.3f)' %(popt[1],np.sqrt(pcov[1,1])))
print('sigma    (%.3f ± %.3f)' %(popt[2],np.sqrt(pcov[2,2])))
print('\n---------------------------------\n')
print('# Chi square test:')
print('Chi2      %.3f' %chi2)
print('dof       %.i'  %dof)
print('Chi2/dof  %.3f' %chi2_rid)
print('pvalue    %.3f' %pvalue)
print('\n---------------------------------\n')
print('FWHM      (%.3f ± %.3f)'   %(FWHM,sigma_FWHM))
print('R         (%.3f ± %.3f)%%' %(R,sigma_R))
print('\n=================================================')


