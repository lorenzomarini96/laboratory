# Fit

## Link

[Numpy - Documentation](https://numpy.org/doc/stable/)

[scipy](https://www.scipy.org)

## Examples

### Least Square algorithm

#### Link

[curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)


 ```python
 # Import packages
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

# Data Simulation
x_data = np.linspace(1,20,10)
y_data = x_data * 1.50 + np.random.normal(0,1,len(x_data))
dy_data = 1.0 * np.ones(len(y_data))

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
a_curve_fit, b_curve_fit             =  popt
sigma_a_curve_fit, sigma_b_curve_fit = np.sqrt(pcov.diagonal())

# Chi2
chi2_lsq     = sum(((y_data - f_curve_fit(x_data, *popt)) / dy_data)**2)
# Numero di gradi di libertà
dof          = len(x_data) - len(p0_curve_fit) 
# Calcolo dei chi2 ridotto
chi2_rid_lsq = chi2_lsq/dof
# Calcolo del p-value
pvalue_lsq   = 1 - stats.chi2.cdf(chi2_lsq, dof) # pvalue deve essere maggiore di 0.05

# Stampa parametri di fit 
print('\n# Fit Function:')
print('f(x) = a * x + b\n')
print(f'a  {a_curve_fit:12.3f} ± {sigma_a_curve_fit:.3f}')
print(f'b  {b_curve_fit:12.3f} ± {sigma_b_curve_fit:.3f}')
print(f'Chi2     {chi2_lsq:12.3f}')
print(f'dof       {dof:12}')
print(f'Chi2/dof  {chi2_rid_lsq:12.3f}')
print(f'pvalue    {pvalue_lsq:12.3f}')

# Grafico 
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
x_fit = np.linspace(min(x_data),max(x_data),100)

# Plot fit 
plt.plot(x_fit, f_curve_fit(x_fit, *popt), '--', color="green", label=r'LSQ')
ax.set_title(r'Least Squares',fontsize=18)
ax.set_xlabel(r'$x$ [$x$]', fontsize=18)
ax.set_ylabel(r'$y$ [$y$]', fontsize=18)
ax.minorticks_on()
# Abbellimenti al plot
ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$a_{LSQ}$  $\>$ %.2f $\pm$ %.2f'   %(a_curve_fit, sigma_a_curve_fit))
ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$b_{LSQ}$  $\>$ %.2f $\pm$ %.2f'   %(b_curve_fit, sigma_b_curve_fit))
ax.plot([], [], color='white', marker='.',linestyle='None', label=r'$\chi^2/\nu$   $\>$ %12.2f/%.i'   %(chi2_lsq,dof))
# Creazione della legenda
ax.legend(frameon = False,fancybox = True, hadow = False,prop = {"size":17},numpoints = 1)
# Salvataggio del grafico in formato .pdf
plt.savefig('figures/lsq__fit.png', bbox_inches='tight')

plt.show()

```

<img src="https://user-images.githubusercontent.com/55988954/104958758-e41b5c00-59d0-11eb-8ae3-4ee90c472a83.png" width="600" /> 
