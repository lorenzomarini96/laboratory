#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()
from scipy.optimize import curve_fit
from scipy import stats

# Define fit function. 
def fit_function(x, A, beta, B, mu, sigma):
    background = A * np.exp(-x/beta)
    signal     = B * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return background + signal
 
# Generate exponential and gaussian data and histogram.
data_background = np.random.exponential(scale=2.0, size=10000)#
data_signal     = np.random.normal(loc=3.0, scale=0.3, size=2000)
bins            = np.linspace(0, 6, 61)
data_entries_background, bin_bordes = np.histogram(data_background, bins=bins)
data_entries_signal, bin_bordes     = np.histogram(data_signal, bins=bins)

# Add histograms of exponential and gaussian data.
data_entries = data_entries_background + data_entries_signal
bin_centers  = 0.5 * (bin_bordes[1:] + bin_bordes[:-1])

# Fit the function to the histogram data.
param0 = [20000, 2.0, 2000, 3.0, 0.3]
popt, pcov = curve_fit(fit_function, xdata=bin_centers, ydata=data_entries, p0=param0)

A, beta, B, mu, sigma = popt
dA, dbeta, dB, dmu, dsigma = np.sqrt(np.diagonal(pcov))

# Chi2 Test
chi2   = sum(((data_entries[data_entries > 0] - fit_function(bin_centers[data_entries > 0], *popt))/np.sqrt(data_entries[data_entries > 0]))**2)
dof    = len(bin_centers) - len(popt) - 1 
pvalue = 1 - stats.chi2.cdf(chi2, dof) # pvalue deve essere maggiore di 0.05

# Full Width at Half Maximum
FWHM = 2 * np.sqrt(2 * np.log(2)) * abs(sigma) 
dFWHM = 2 * np.sqrt(2 * np.log(2)) * dsigma
# Risuoluzione
R = (FWHM/mu) * 100 # Energy resolution = (FWHM /photo peak) x 100
dR = R * (np.sqrt((dFWHM/FWHM)**2 + (dmu/mu)**2)) # error propagation

popt_list = ["A    ", "beta ", "B    ", "mu   ", "sigma"]
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
print("==============================")
print("FWHM and Resolution\n")
print(f"FWHM {FWHM:12.5f} +/- {dFWHM:.5}")
print(f"R    {R:12.5f} +/- {dR:.5}")
print("==============================")

# Plot the fitting function.
fig = plt.figure(1, figsize=(10,8))
ax1 = fig.add_axes([0,0,1,1])
ax1.errorbar(bin_centers, data_entries, yerr=np.sqrt(data_entries),
             fmt='.', 
             color='black', 
             label =r'data'
             )

x_fit = np.linspace(min(bin_centers), max(bin_centers), 10000)
ax1.plot(x_fit, fit_function(x_fit, *(popt)), '-', color="red")

# Ridefine functions to make the fits
def signal(x, B, mu, sigma):
    return B * np.exp(-(x - mu)**2 / (2 * sigma**2))

def background(x, A, beta):
    return A * np.exp(-x/beta)

ax1.plot(x_fit, signal(x_fit, B, mu, sigma), '-', color="blue", label=r'signal = $B e^{-(x - \mu)^2/2\sigma^2}$')
ax1.plot(x_fit, background(x_fit, A, beta), '--', color="green",label='background = $A e^{-x / \\beta}$')

# Make the plot nicer.
ax1.set_ylim(0)
#ax1.set_xlabel(r"\textbf{bins values}", fontsize=14)
ax1.set_ylabel(r"Number of entries",          fontsize=14)
ax1.set_title(r"Signal Peak over Background", fontsize=14)
ax1.minorticks_on()
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'$A$      {A:12.2f} $\pm$ {dA:.2f}')
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'$\beta$  {beta:12.2f} $\pm$ {dbeta:.2f}')
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'$\mu$    {mu:12.2f} $\pm$ {dmu:.2f}')
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'$\sigma$ {sigma:12.2f} $\pm$ {dsigma:.2f}')
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'$\chi^2$ {chi2:12.2f}/{dof}')
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'p        {pvalue:12.2f}')
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'FWHM     {FWHM:5.2f} $\pm$ {dFWHM:.2f}')
ax1.plot([], [], color='white', marker='.',linestyle='None', label=rf'$\frac{{\Delta x}}{{x}}$      {R:12.2f} $\pm$ {dR:.2f}')
ax1.legend(fancybox=True, shadow=False, loc='best', prop={"size":16}, numpoints = 1)

# Residual plot
ax2     = fig.add_axes([0.0,-0.3,1,0.3])
r       = data_entries - fit_function(bin_centers, *popt)
sigma_r = np.sqrt(data_entries)
ax2.plot(bin_centers, np.zeros(len(data_entries)), '--', color="red")
ax2.errorbar(bin_centers, r, sigma_r, linestyle = '', color = 'black', marker = 'o', label=r'Error')
ax2.set_xlabel(r'$x$ [x]',        fontsize=14)
ax2.set_ylabel(r'$Residuals$',fontsize=14)
ax2.minorticks_on()
plt.savefig('figures/Signal_Peak_over_Background1.png', format='png',bbox_inches="tight", dpi=100)
#plt.show()