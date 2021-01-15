""" Script that implements an example of an LSQ algorithm applied on a random data set."""

# Package for data management and fit.
import pandas as pd
import numpy as np
import random

# 1) Package for least squares (LSQ) fit.
from scipy.optimize import curve_fit

# 2) Package to compute p-value.
from scipy import stats

# Package for make plots
import matplotlib
import matplotlib.pyplot as plt
# for LaTeX
from matplotlib import rcParams
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)


#==========================================
# Data Simulation
#==========================================

x_data = np.linspace(1,20,10)
y_data = x_data * 1.50 + np.random.normal(0,1,len(x_data))
dy_data = 1.0 * np.ones(len(y_data))
# Make dataframe for simulated data
data = {'x_data': x_data, 'y_data': y_data, 'dy_data': dy_data}
# Make a table of data
table = pd.DataFrame(data)
print(table)


def least_squares(x_data, y_data, dy_data):
    """
    Function implementing the least squares algorithm on given data.

    Parameters
    ----------
    x_data : array
           Description
    y_data : array
           Description
    dx_data : array
           Description
    dy_data : array
           Description

    Returns
    -------
    None
    """
    
    ################################################################
    # CURVE FIT 
    ################################################################
    
    # Funzione modello (teorico)
    def f_curve_fit(x, a, b):
        return a * x + b
    
    p0_curve_fit  = [1, 1]           # valori iniziali dei parametri

    popt, pcov    = curve_fit(
        f         = f_curve_fit,     # funzione modello 
        xdata     = x_data,          # x data
        ydata     = y_data,          # y data
        sigma     = dy_data,         # incertezza su y
        p0        = p0_curve_fit     # valori iniziali dei parametri
) 
    
    # Parametri di Best Fit 
    a_curve_fit       = popt[0]
    sigma_a_curve_fit = np.sqrt(pcov[0,0])
    b_curve_fit       = popt[1]
    sigma_b_curve_fit = np.sqrt(pcov[1,1])

    #=========================================
    # Coefficiente di determinazione R^2 - LSQ
    #=========================================
    R2_lsq = np.sum((f_curve_fit(x_data, *popt) - y_data.mean())**2) / np.sum((y_data - y_data.mean())**2)
    #print("R^2 (LSQ) %12.3f" % R2_lsq)
    
    #=========================================
    # Test X2 - LSQ
    #=========================================
    # Chi2
    chi2_lsq     = sum(((y_data - f_curve_fit(x_data, *popt)) / dy_data)**2)
    # Numero di gradi di libertà
    dof          = len(x_data) - len(p0_curve_fit) 
    # Calcolo dei chi2 ridotto
    chi2_rid_lsq = chi2_lsq/dof
    # Calcolo del p-value
    pvalue_lsq   = 1 - stats.chi2.cdf(chi2_lsq, dof) # pvalue deve essere maggiore di 0.05

    print('\n###################################################')
    print('Least Squares (LSQ)')
    print('###################################################')
    # Stampa parametri di fit 
    print('\n# Fit Function:')
    print('f(x) = a * x + b\n')
    print('# Parametri del fit (LSQ):')
    print(f'a  {a_curve_fit:12.3f} ± {sigma_a_curve_fit:.3f}')
    print(f'b  {b_curve_fit:12.3f} ± {sigma_b_curve_fit:.3f}')
    print(f'Chi2     {chi2_lsq:12.3f}')
    print(f'dof       {dof:12}')
    print(f'Chi2/dof  {chi2_rid_lsq:12.3f}')
    print(f'pvalue    {pvalue_lsq:12.3f}')
    
    #=========================================
    # Grafico - LSQ
    #=========================================
    fig = plt.figure(1, figsize=(10,6))
    ax  = plt.axes()
  
    ax.errorbar(x_data,  y_data,
                 yerr = dy_data,
                 linestyle = '',
                 color     = 'black', 
                 ecolor    = 'black',
                 marker    = 'o',
                 capsize   = 3,
                 #label     = r'data'
               )
    
    # Definizione vettore delle x (asse x di estremi minimo e massimo di x_data)
    x = np.linspace(min(x_data),max(x_data),100)

    # Plot fit 
    plt.plot(x, f_curve_fit(x, *popt),
         '--', color = "green", 
         label       = r'LSQ'
        )
    ax.set_title(r'Least Squares',fontsize=18)
    ax.set_xlabel(r'$x$ [$x$]', fontsize=18)
    ax.set_ylabel(r'$y$ [$y$]', fontsize=18)
    ax.minorticks_on()

    # Abbellimenti al plot
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_{LSQ}$  $\>$ %.2f $\pm$ %.2f'   %(a_curve_fit, sigma_a_curve_fit))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$b_{LSQ}$  $\>$ %.2f $\pm$ %.2f'   %(b_curve_fit, sigma_b_curve_fit))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$\chi^2/\nu$   $\>$ %12.2f/%.i'   %(chi2_lsq,dof))
    # Stampo il simbolo dell'elemento sopra ogni valore nel plot 
    for i in range(len(y_data)):
        plt.text(y_data[i], x_data[i], y_data[i], fontsize=15)
    # Creazione della legenda
    ax.legend(frameon        = False,
              fancybox       = True, 
              shadow         = False,
              #loc            = 'center left',
              #bbox_to_anchor = (1, 0.5),
              prop           = {"size":17},
              numpoints      = 1)

    # Salvataggio del grafico in formato .pdf
    #plt.savefig('lsq__fit.pdf', bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    least_squares(x_data, y_data, dy_data)
