""" Script that implements an example of an ODR algorithm applied on a random data set."""

# Package for data management and fit.
import pandas as pd
import numpy as np
import random

# 1) Package for least squares (LSQ) fit.
from scipy.optimize import curve_fit

# 2) Package for orthogonal distance regression (ODR) fit.
from scipy.odr import ODR, Model, Data, RealData

# 3) Package to compute p-value.
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
dx_data = 0.50 * np.ones(len(x_data))
dy_data = 0.50 * np.ones(len(y_data))
# Make dataframe for simulated data
data = {'x_data': x_data, 'y_data': y_data, 'dx_data': dx_data, 'dy_data': dy_data}
# Make a table of data
table = pd.DataFrame(data)
print(table)


#==========================================
# Orthogonal Distance Regression (ODR) fit
#==========================================

def orthogonal_distance_regression(x_data, y_data, dx_data, dy_data):
    """
    Function implementing the orthogonal distance regression algorithm on given data.

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

    #==========================================
    # ODR algorithm
    #==========================================
    
    def f_odr(beta, x):
        '''
        Function to estimate the optimal parameters
        (attention: not usable for the lot)
        '''
        a, b = beta 
        return a * x + b
    
    # Data definition
    data                     = RealData(x_data, y_data, dx_data, dy_data)
    # Model definition
    model                    = Model(f_odr)
    p0                       = (1.0,1.0)
    odr                      = ODR(data, model, p0)
    odr.set_job(fit_type=0)
    odr_output               = odr.run()
    a_odr,       b_odr       = odr_output.beta
    sigma_a_odr, sigma_b_odr = np.sqrt(np.diag(odr_output.cov_beta))
    
    #==========================================
    # Model function (for the plot)
    #==========================================

    def f_fit(x, a, b):
        '''
        Funzione utilizzata per il plot vero e proprio 
        '''
        return a * x + b
    
    #==========================================
    # Coefficiente di determinazione R^2 - ODR
    #==========================================
    R2_odr = np.sum((f_fit(x_data, *odr_output.beta) - y_data.mean())**2) / np.sum((y_data - y_data.mean())**2)
    
    #==========================================
    # Test X2 - ODR
    #==========================================
    # Chi2
    #chi2    = sum(((y_data - f_fit(x_data, a_odr, b_odr)) / dy_data)**2)
    chi2_odr     = odr_output.sum_square
    # Numero di gradi di libertà
    dof          = len(x_data) - len(p0) 
    # Calcolo dei chi2 ridotto
    chi2_rid_odr = chi2_odr/dof
    # Calcolo del p-value
    pvalue_odr   = 1 - stats.chi2.cdf(chi2_odr, dof) # pvalue deve essere maggiore di 0.05
    
    print('#==========================================')
    print('# Orthogonal Distance Regression (ODR) fit')
    print('#==========================================')
    # Stampa parametri di fit 
    print('\n# Fit Function:')
    print('f(x) = a * x + b\n')
    print('# Parametri del fit (ODR):')
    print(f'a  {a_odr:12.3f} ± {sigma_a_odr:3f}')
    print(f'b  {b_odr:12.3f} ± {sigma_b_odr:3f}')    
    print(f'\nR2 (ODR)  {R2_odr:12.3f}')
    print(f'Chi2      {chi2_odr:12.3f}')
    print(f'dof       {dof}')
    print(f'Chi2/dof  {chi2_rid_odr:12.3f}')
    print(f'pvalue    {pvalue_odr:12.3f}')

    #=========================================
    # Grafico - ODR
    #=========================================

    fig = plt.figure(1, figsize=(10,6))
    ax  = plt.axes()
    ax.errorbar(x_data,  y_data,
                 xerr = dx_data, yerr = dy_data,
                 linestyle = '',
                 color     = 'black', 
                 ecolor    = 'black',
                 marker    = 'o',
                 capsize   = 3,
                 #label     = r'data'
               )
    # Definizione vettore delle x (asse x di estremi minimo e massimo di x_data)
    x_fit = np.linspace(min(x_data),max(x_data),100)

    # Plot fit 
    ax.plot(x_fit, f_fit(x_fit, *odr_output.beta),
         '--', color = "red", 
         label       = r'ODR'
        )
    ax.set_title(r'Orthogonal Distance Regression',fontsize=18)
    ax.set_xlabel(r'$x$ [$x$]', fontsize=18)
    ax.set_ylabel(r'$y$ [$y$]', fontsize=18)
    ax.minorticks_on()
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_{ODR}$  $\>$ %12.2f $\pm$ %.2f'   %(a_odr, sigma_a_odr))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$b_{ODR}$  $\>$ %12.2f $\pm$ %.2f'   %(b_odr, sigma_b_odr))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$\chi^2/\nu$   $\>$ %12.2f/%.i'   %(chi2_odr,dof))
    # Creazione della legenda
    ax.legend(frameon        = False,
              fancybox       = True, 
              shadow         = False,
              #loc            = 'center left',
              #bbox_to_anchor = (1, 0.5),
              prop           = {"size":17},
              numpoints      = 1)
    plt.savefig('odr__fit.pdf', bbox_inches='tight')
    plt.show()


def least_squares(x_data, y_data, dx_data, dy_data):
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
                 xerr = dx_data, yerr = dy_data,
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

    # Creazione della legenda
    ax.legend(frameon        = False,
              fancybox       = True, 
              shadow         = False,
              #loc            = 'center left',
              #bbox_to_anchor = (1, 0.5),
              prop           = {"size":17},
              numpoints      = 1)

    # Salvataggio del grafico in formato .pdf
    plt.savefig('lsq__fit.pdf', bbox_inches='tight')

    plt.show()


#==========================================
# ODR vs LSQ fit
#==========================================

def odr_vs_lsq(x_data, y_data, dx_data, dy_data):
    """
    Function implementing the orthogonal distance regression algorithm on given data.

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

    #==========================================
    # ODR algorithm
    #==========================================
    
    def f_odr(beta, x):
        '''
        Function to estimate the optimal parameters
        (attention: not usable for the lot)
        '''
        a, b = beta 
        return a * x + b
    
    # Data definition
    data                     = RealData(x_data, y_data, dx_data, dy_data)
    # Model definition
    model                    = Model(f_odr)
    p0                       = (1.0,1.0)
    odr                      = ODR(data, model, p0)
    odr.set_job(fit_type=0)
    odr_output               = odr.run()
    a_odr,       b_odr       = odr_output.beta
    sigma_a_odr, sigma_b_odr = np.sqrt(np.diag(odr_output.cov_beta))
    
    #==========================================
    # Model function (for the plot)
    #==========================================

    def f_fit(x, a, b):
        '''
        Funzione utilizzata per il plot vero e proprio 
        '''
        return a * x + b
    
    #==========================================
    # Coefficiente di determinazione R^2 - ODR
    #==========================================
    R2_odr = np.sum((f_fit(x_data, *odr_output.beta) - y_data.mean())**2) / np.sum((y_data - y_data.mean())**2)
    
    #==========================================
    # Test X2 - ODR
    #==========================================
    # Chi2
    #chi2    = sum(((y_data - f_fit(x_data, a_odr, b_odr)) / dy_data)**2)
    chi2_odr     = odr_output.sum_square
    # Numero di gradi di libertà
    dof          = len(x_data) - len(p0) 
    # Calcolo dei chi2 ridotto
    chi2_rid_odr = chi2_odr/dof
    # Calcolo del p-value
    pvalue_odr   = 1 - stats.chi2.cdf(chi2_odr, dof) # pvalue deve essere maggiore di 0.05
    
    print('#==========================================')
    print('# Orthogonal Distance Regression (ODR) fit')
    print('#==========================================')
    # Stampa parametri di fit 
    print('\n# Fit Function:')
    print('f(x) = a * x + b\n')
    print('# Parametri del fit (ODR):')
    print(f'a  {a_odr:12.3f} ± {sigma_a_odr:3f}')
    print(f'b  {b_odr:12.3f} ± {sigma_b_odr:3f}')    
    print(f'\nR2 (ODR)  {R2_odr:12.3f}')
    print(f'Chi2      {chi2_odr:12.3f}')
    print(f'dof       {dof}')
    print(f'Chi2/dof  {chi2_rid_odr:12.3f}')
    print(f'pvalue    {pvalue_odr:12.3f}')

    
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
    dof          = len(x_data) - len(p0) 
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
    # Grafico - ODR
    #=========================================

    fig = plt.figure(1, figsize=(10,6))
    ax  = plt.axes()
  
    ax.errorbar(x_data,  y_data,
                 xerr = dx_data, yerr = dy_data,
                 linestyle = '',
                 color     = 'black', 
                 ecolor    = 'black',
                 marker    = 'o',
                 capsize   = 3,
                 #label     = r'data'
               )
    
    # Definizione vettore delle x (asse x di estremi minimo e massimo di x_data)
    x_fit = np.linspace(min(x_data),max(x_data),100)

    # Plot fit 
    ax.plot(x_fit, f_fit(x_fit, *odr_output.beta),
         '--', color = "red", 
         label       = r'ODR'
        )
    ax.set_title(r'ODR vs LSQ',fontsize=18)
    ax.set_xlabel(r'$x$ [$x$]', fontsize=18)
    ax.set_ylabel(r'$y$ [$y$]', fontsize=18)
    ax.minorticks_on()
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_{ODR}$  $\>$ %12.2f $\pm$ %.2f'   %(a_odr, sigma_a_odr))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$b_{ODR}$  $\>$ %12.2f $\pm$ %.2f'   %(b_odr, sigma_b_odr))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$\chi^2/\nu$   $\>$ %12.2f/%.i'   %(chi2_odr,dof))

    #=========================================
    # Grafico - LSQ
    #=========================================
    
    # Definizione vettore delle x (asse x di estremi minimo e massimo di x_data)
    x = np.linspace(min(x_data),max(x_data),100)

    # Plot fit 
    plt.plot(x, f_curve_fit(x, *popt),
         '--', color = "green", 
         label       = r'LSQ'
        )

    # Abbellimenti al plot
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_{LSQ}$  $\>$ %.2f $\pm$ %.2f'   %(a_curve_fit, sigma_a_curve_fit))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$b_{LSQ}$  $\>$ %.2f $\pm$ %.2f'   %(b_curve_fit, sigma_b_curve_fit))
    ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$\chi^2/\nu$   $\>$ %12.2f/%.i'   %(chi2_lsq,dof))
    # Creazione della legenda
    ax.legend(frameon        = False,
              fancybox       = True, 
              shadow         = False,
              #loc            = 'center left',
              #bbox_to_anchor = (1, 0.5),
              prop           = {"size":17},
              numpoints      = 1)
    plt.savefig('odr_vs_lqs_fit.pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    orthogonal_distance_regression(x_data, y_data, dx_data, dy_data)
    least_squares(x_data, y_data, dx_data, dy_data)
    odr_vs_lsq(x_data, y_data, dx_data, dy_data)
