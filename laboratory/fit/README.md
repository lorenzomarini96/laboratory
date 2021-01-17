# Fit

## Link
[Numpy][]
[scipy][]
[curve_fit][]
## Examples

### Least Square algorithm
#### Link
[curve_fit][]

 ```
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
print('# Parametri del fit (LSQ):')
print(f'a  {a_curve_fit:12.3f} ± {sigma_a_curve_fit:.3f}')
print(f'b  {b_curve_fit:12.3f} ± {sigma_b_curve_fit:.3f}')
print(f'Chi2     {chi2_lsq:12.3f}')
print(f'dof       {dof:12}')
print(f'Chi2/dof  {chi2_rid_lsq:12.3f}')
print(f'pvalue    {pvalue_lsq:12.3f}')
    
```