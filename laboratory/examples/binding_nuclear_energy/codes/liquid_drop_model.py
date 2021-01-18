import numpy as np
import scipy
import pandas as pd
 
import pylab
from scipy.optimize import curve_fit 
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

# Symbol 
nucleus = ['H', 'He', 'He', 'Li', 'Li', 'Be', 'B', 'B',
           'C', 'C', 'N', 'N', 'O', 'O', 'O', 'F', 'Ne', 'Ne', 'Ne',
          'Na', 'Mg', 'Mg', 'Al', 'Si', 'Si', 'P', 'S', 'Cl', 'Ar', 'Ar',
           'K', 'Ca', 'Ca', 'Sc', 'Ti', 'V*', 'Cr', 'Mn', 'Fe', 'Fe', 'Fe',
          'Co', 'Ni', 'Ni', 'Cu', 'Sn', 'Gd', 'Yb', 'Hg', 'Th', 'U']

Z      = np.array([1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 10,
         10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 17, 18, 18, 19,20,
         20, 21, 22, 23, 24, 25, 26, 26, 26, 27, 28, 28, 29, 50, 64, 70, 80, 90, 92])

A      = np.array([3, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 25, 26, 27, 29, 30, 31, 34, 37, 38, 40, 41, 43,
         44, 45, 48, 50, 52, 55, 56, 57, 58, 59, 61, 62, 63, 115, 156, 173, 204, 232, 238])

N      = np.array([2, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 10,
         11, 12, 12, 13, 14, 14, 15, 16, 16, 18, 20, 20, 22, 22, 23,
         24, 24, 26, 27, 28, 30, 30, 31, 32, 32, 33, 34, 34, 65, 92, 193, 124, 142, 146])

B_over_A_exp = np.array([2.827, 2.57267, 7.074, 5.33233,
                5.60629, 6.46278, 6.4751, 6.92773, 7.68017, 7.46985, 7.47564, 
                7.69947, 7.97619, 7.75076, 7.76706, 7.779, 8.03225,
                7.97171, 8.08045, 8.11148, 8.22352, 8.33388, 8.33156, 8.44866,
                8.52067, 8.48119, 8.5835, 8.5703, 8.61429, 8.59528, 8.57607,
                8.60067, 8.65818, 8.61884, 8.72292, 8.69588, 8.77594, 8.765, 8.79032,
                8.77026, 8.79222, 8.76802, 8.76502, 8.79455, 8.75214, 8.5141, 8.2133, 8.08746, 7.88555, 7.61503,
                         7.57013])
sigma_B_over_A = 0.03*B_over_A_exp


fig = plt.figure(1, figsize = (15,6))
plt.rc('font', size=16)
ax1 = fig.add_axes([0,0,1,1])
ax1.errorbar(A, B_over_A_exp, yerr=sigma_B_over_A,
             fmt='.', 
             color='black', 
             label =r'data'
             )
# Funzione di fit 
def liquid_drop_model(A, a_v, a_s, a_c, a_sym, a_p):
    """Liquid drop model implemented on binding energy data.
    
    Parameters
    ----------
    A : int
        Atomic mass number
    a_v: float
        Volume term
    a_s: float
        Surface term
    a_c: float
        Coulomb term
    a_sym: float
        Pauli term
    a_p: float
        Pairing term


    Returns
    -------
    float
        Binding energy for nucleon computed via the Weizsäcker formula.
    
    """
    # Termine di Pairing 
    if (Z.any() % 2) == 0 and (N.any() % 2) == 0:  # Se N,Z pari - pari
        delta = a_p / A**(7/4)
    if (Z.any() % 2) != 0 or (N.any() % 2) == 0:   # Se N,Z dispari - pari (o viceversa)
        delta = 0
    if (Z.any() % 2) == 0 or (N.any() % 2) != 0:   # Se N,Z dispari - pari (o viceversa)
        delta = 0
    if (Z.any() % 2) != 0 and (N.any() % 2) != 0:   # Se N,Z dispari - dispari
        delta = - a_p / A**(7/4)
        
    # Binding Energy per nucleon
    B_over_A_model = a_v - a_s/A**(1./3.) - a_c * Z*(Z-1)/A**(4./3.) - a_sym*((A - 2*Z)/A)**2 - delta
    
    return B_over_A_model

# Parametri iniziali
param0 = [15.68, 18.56, 0.717, 28.1, 34.0]
# Best Parameters
popt, pcov = curve_fit(liquid_drop_model, A, B_over_A_exp, param0, sigma =sigma_B_over_A)
a_v, a_s, a_c, a_sym, a_p = popt
sigma_a_v, sigma_a_s, sigma_a_c, sigma_a_sym, sigma_a_p = np.sqrt(np.diagonal(pcov))

# Chi2 Test
chi2   = sum(((B_over_A_exp - liquid_drop_model(A, *popt))/sigma_B_over_A)**2)
dof    = len(A) - len(popt)
pvalue = 1 - stats.chi2.cdf(chi2, dof)

popt_list = ["a_v  ", "a_s  ", "a_c  ", "a_sym", "a_p  "]
print("==============================")
print("Best fit result:\n")
[print(f'{popt_list[i]} {popt[i]:12.5f} +/- {np.sqrt(pcov[i][i]):.5f}') for i in range(len(popt_list))]
print("==============================")
print("==============================")
print('# Chi square test\n')
print(f"chi2     {chi2:12.3f}")
print(f"dof      {dof:12}")
print(f"chi2/dof {chi2:12.3f}/{dof}")
print(f"pvalue   {pvalue:12.3f}")
print("==============================")

# Plot fit
ax1.plot(A, liquid_drop_model(A, *popt), '--', color="red", label=r'$B/A$ = $a_v A - a_s A^{2/3} - a_c \frac{Z(Z-1)}{A^{1/3}} - a_{sym}\frac{(N - Z)^2}{A^2} + \frac{\delta(A,Z)}{A}$')
ax1.set_title('Binding Energy per nuclear particle (nucleon)') 
ax1.set_xlabel(r'A') 
ax1.set_ylabel(r'B/A [MeV]')
ax1.minorticks_on()
ax1.set_ylim(0)
ax1.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_v$     $\>$  $\>$ %.2f $\pm$ %.2f'   %(a_v, sigma_a_v))
ax1.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_s$     $\>$  $\>$ %.2f $\pm$ %.2f'   %(a_s, sigma_a_s))
ax1.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_c$     $\>$ $\>$  %.2f $\pm$ %.2f'   %(a_c, sigma_a_c))
ax1.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_{sym}$ $\>$ %.2f  $\pm$ %.2f'   %(a_sym, sigma_a_sym))
ax1.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_p$     $\>$$\>$   %.2f $\pm$ %.2f'   %(a_p, sigma_a_p))
ax1.plot([], [], color='white', marker='.',linestyle='None', label=r'$\chi^2$/dof  $\>$   %.2f/%.i'   %(chi2,dof))
ax1.plot([], [], color='white', marker='.',linestyle='None', label=r'$p_{value}$ $\>$     %.2f'   %pvalue)
ax1.legend(frameon = False,fancybox=True,loc='best',  numpoints = 1, fontsize=16)
# Residual plot
ax2     = fig.add_axes([0.0,-0.3,1,0.3])
r       = B_over_A_exp - liquid_drop_model(A, *popt)
sigma_r = sigma_B_over_A
ax2.plot(A, np.zeros(len(B_over_A_exp)), '--', color="red")
ax2.errorbar(A, r, sigma_B_over_A, linestyle = '', color = 'black', marker = 'o', label=r'Error')
ax2.set_xlabel(r'A',        fontsize=16)
ax2.set_ylabel(r'$Residuals$',fontsize=16)
ax2.minorticks_on()
plt.savefig('../figures/B_over_A_light.png',format='png',bbox_inches="tight",dpi=100) 
plt.show()