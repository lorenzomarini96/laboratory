# 3D

import random
import numpy as np
import math
from colorama import Fore, Style

np.random.seed(1000)

# 1) INIZIALIZZAZIONE 
# Costanti fisiche

A        =  12           # Numero di massa del nucleo diffusore (Carbonio 12) (necessario per il calcolo delle traiettorie)
Sigma_a  =  2.738e-4     # Sezione d'urto macroscopica di assorbimento [cm^-1]
Sigma_el =  0.3851       # Sezione d'urto macroscopica di scattering elastico [cm^-1]
v        =  2.2e5        # Modulo della Velocità dei neutroni [cm/s] 
N        =  1            # Numero di neutroni da simulare (# eventi)
n_mass   =  939.565      # Massa del neutrone [MeV/c²] 

#====================================================

Sigma_t  = Sigma_a + Sigma_el   # Sezione d'urto macroscopica totale [cm^-1]
iSigma_t = 1/Sigma_t            # Libero cammino medio (tra due interazioni successive) [cm]


# Probabilità di assorbimento del neutrone
p_a   = Sigma_a/Sigma_t
# Probabilità di essere diffuso (scattering elastico)    
p_el  = Sigma_el/Sigma_t

# Probabilità di assorbimento del neutrone in %
p_a_perc   = p_a * 100
# Probabilità di essere diffuso (scattering elastico) in %
p_el_perc  = p_el * 100

'''
print ("\nProbabilità assorbimento: {} %".format(p_a_perc))
print ("\nProbabilità scattering: {} %".format(p_el_perc))
'''
#====================================================
# Vettori (per adesso liste) da istogrammare
d_list = []    

# Vettore del numero delle collisioni di ciascun neutrone 
N_coll_vec = np.zeros(N)

# Vettore della distanza TOTALE percorsa dal neutrone prima di essere assorbito da un nucleo 
d_tot_vec  = np.zeros(N)

# Vettore della distanza tra sorgente - punto finale di arresto
r_vec      = np.zeros(N)

# Vettore dei tempi di volo 
t_vec      = np.zeros(N)

#====================================================

for i in range(0,N):
    
    # Liste coordinate spaziali per ciascun neutrone
    x_list       = []
    y_list       = []
    z_list       = []
       
    #====================================================

    # 2 CINEMATICA
     
    # Direzione iniziale - Angoli di emissione del neutrone
    xi    = random.uniform(0,1)                
    theta = 57.2958 * math.acos(1 - 2 * xi)                            # Angolo polare in gradi °
    phi   = 57.2958 * 2 * np.pi * xi                                   # Angolo azimutale in gradi °
    '''
    print('\n# Direzione di emissione SCM ')
    print('theta = %.3f °' %theta)
    print('phi   = %.3f °' %phi)  
    '''
    # Coseni direttori 
    alpha = math.sin(math.radians(theta))*math.cos(math.radians(phi)) 
    beta  = math.sin(math.radians(theta))*math.sin(math.radians(phi))
    gamma = math.cos(math.radians(theta))
    
    '''
    print('\n# Coseni direttori')
    print('alpha  = %.3f °' %alpha)
    print('beta   = %.3f °' %beta)        
    print('gamma  = %.3f °' %gamma)
    '''
    #====================================================

    # 3 TRACCIAMENTO
    
    # Posizione iniziale [cm]
    x = 0.000
    y = 0.000
    z = 0.000
   
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
        
    x_vec = np.asarray(x_list)
    y_vec = np.asarray(y_list)
    z_vec = np.asarray(z_list)
    
    # Inizializza il valore delle variabili:
    # Distanza totale percorsa dal neutrone [cm]
    d_tot = 0.000 

    # Numero di collisioni
    N_coll = 0   
    
    # Tempo di volo [s]
    t_volo_tot = 0.000
    

    # Il neutrone percorre un certo tratto d (Campionamento libero cammino medio)
    xi_1  = np.random.uniform(0,1)
    d     = - iSigma_t * np.log(xi_1) # Lunghezza tra il punto di partenza e quello e quello dove avviene la collisione successiva.
    
    '''
    print('\n# Distanza d')
    print('d = %.4f cm' %d)
    '''
    
    # Nuova posizione 
    x = x + d * alpha
    y = y + d * beta
    z = z + d * gamma
    
    #====================================================

    # 4) DESTINO DEL NEUTORNE
    
    xi_2   = np.random.uniform(0,1)
           
    while(xi_2 > p_a):
        '''
        print(Fore.BLUE + '\n\nn_%i scatterato --> ' %i)
        print(Style.RESET_ALL)
        '''
        
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        
        x_vec = np.asarray(x_list)
        y_vec = np.asarray(y_list)
        z_vec = np.asarray(z_list)
        
        '''
        print('# Coordinate del neutrone')
        print('x_n =', x_vec.round(decimals=4))
        print('y_n =', y_vec.round(decimals=4))
        print('z_n =', z_vec.round(decimals=4))
        '''
        
        # NEUTRONE SCATTERATO
        # Calcolo nuovi angoli di emissione del neutrone scatterato (nel sistema del CM)
        # Nuova direzione di emissione - Angoli di emissione del neutrone
        xi       = random.uniform(0,1)                
        theta_cm = 57.2958 * math.acos(1 - 2 * xi)          # Angolo polare in gradi °
        phi_cm   = 57.2958 * 2 * np.pi * xi                 # Angolo azimutale in gradi °

        '''
        print('\n# Nuova direzione di emissione SCM ')
        print('theta_cm = %.3f °' %theta_cm)
        print('phi_cm   = %.3f °' %phi_cm) 
        '''
            
        # Trasformazione nel sistema del LAB 
        cos_theta  = (1 + A * math.cos(math.radians(theta_cm))) / (np.sqrt(A**2 + 2 * A * math.cos(math.radians(theta_cm)) + 1))
        theta      = 57.2958 * math.acos(cos_theta)
        phi        = phi_cm
        
        '''
        print('\n# Trasformazione nel sistema del LAB ')
        print('cos_theta_new = %.3f °' %cos_theta)
        print('theta_new     = %.3f °' %theta)        
        print('phi_new       = %.3f °' %phi)
        '''
  
        # Coseni direttori nuovi 
        alpha = math.sin(math.radians(theta))*math.cos(math.radians(phi)) 
        beta  = math.sin(math.radians(theta))*math.sin(math.radians(phi))
        gamma = math.cos(math.radians(theta))
        
        '''
        print('\n# Coseni direttori NEW')
        print('alpha_new = %.3f °' %alpha)
        print('beta_new  = %.3f °' %beta)        
        print('gamma_new = %.3f °' %gamma)
        '''

        # Destino del neutrone scatterato 
        xi_2  = np.random.uniform(0,1)    # Riestraggo un nuovo numero random 
        d     = - iSigma_t * np.log(xi_2) # d: passo della simulazione = lunghezza tra il punto di partenza e quello e quello dove avviene la collisione successiva.
    
        '''
        print('\n# Distanza percorsa d')
        print('d = %.4f cm' %d)
        '''
        
        # Nuova posizione 
        x = x + d * alpha
        y = y + d * beta
        z = z + d * gamma
        
        # Distanza totale percorsa dal neutrone (sarà la somma dei singoli spostamenti dopo gli scattering)
        d_tot       = d_tot + d
        
        # Incremento il numero di collisioni
        N_coll      = N_coll + 1
       
        # Tempo di volo totale del neutrone 
        t_volo_tot  = t_volo_tot + d/v
          
        
        # PRINT:
        '''
        print('\n# Distanza percorsa d')
        print('d_tot     = %.4f cm'  %d_tot)
        print('\n# Tempo di diffusione del neutrone')
        print('t_volo_tot = %.8f s'  %t_volo_tot)
        print('\n# Numero collisioni (elastiche) subite')
        print('N_coll     = %.i'     %N_coll)
        '''
        
                
    # ASSORBIMENTO (xi_2 <= p_a):
    
    # Distanza tra sorgente - punto finale di arresto
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Numero di collisioni (elastiche)
    N_coll_tot = N_coll
    
    '''        
    print(Fore.RED +'\n\nn_%i assorbito: FINE' %i)
    print(Style.RESET_ALL)
    print('# Coordinate del nucleo ASSORBITORE')  
    print('xa = %.3f cm'   %x)
    print('ya = %.3f cm'   %y)
    print('za = %.3f cm\n' %z)
    '''
    
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
        
    x_vec = np.asarray(x_list)
    y_vec = np.asarray(y_list)
    z_vec = np.asarray(z_list)
            
    #====================================================
    '''    
    print('# STORIA DEL NEUTRONE_%.i' %i)
            
    print('# Coordinate del neutrone')
    print('x_n =', x_vec.round(decimals=4))
    print('y_n =', y_vec.round(decimals=4))
    print('z_n =', z_vec.round(decimals=4))
        
    print('\n# Distanza totale percorsa') # Ovviamente sarà maggiore di r (?)
    print('d_tot = %.3f cm'     %d_tot )

    print('\n# Distanza tra sorgente e punto finale di arresto')
    print('r = %.3f cm'         %r)
        
    print('\n# Numero collisioni (elastiche) subite')
    print('N_coll = %.i'        %N_coll)
            
    print('\n# Tempo di volo totale')
    print('t_volo_tot = %.3e s' %t_volo_tot)
    
    '''
    #====================================================

    # 4) Aggiornamento delle componenti dei vettori da istogrammare
    N_coll_vec[i] = N_coll_vec[i] + N_coll
    d_tot_vec[i]  = d_tot_vec[i]  + d_tot
    r_vec[i]      = r_vec[i]      + r
    t_vec[i]      = t_vec[i]      + t_volo_tot 

'''
print(Fore.BLUE + '\n====================================================')
print('# RISULTATI DELLA SIMULAZIONE')
print('====================================================')
print(Style.RESET_ALL)
print('\n# Distanza totale percorsa [cm]') # Ovviamente sarà maggiore di r (?)
print('d_tot_vec  = ',d_tot_vec)            
print('\n# Distanza tra sorgente e punto finale di arresto [cm]')
print('r_vec      = ',r_vec)
print('\n# Numero collisioni (elastiche) subite')
print('N_coll_vec = ',N_coll_vec)  
print('\n# Tempo di volo totale [s]')
print('t_vec      = ',t_vec)  
#print('\n# Angolo di scattering [°]')
#print('theta_vec  = ',theta_vec) 
print(Fore.BLUE +'\n====================================================\n')
print(Style.RESET_ALL)
'''

#====================================================

# Traiettoria 3D
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('classic')
matplotlib.rc('font', family='serif', size=14)
from mpl_toolkits.mplot3d import Axes3D

# Coordinate sorgente di neutroni
x_s = ([0.0])
y_s = ([0.0])
z_s = ([0.0])

# Coordinate punto di assorbimento 
x_stop = x_vec[len(x_vec)-1]
y_stop = y_vec[len(y_vec)-1]
z_stop = z_vec[len(z_vec)-1]

#====================================================
# Per la colorbar (calcolo del vettore delle distanze)
r_vettore = np.zeros(len(x_vec))
for i in range(len(x_vec)):
    
    r_vettore[i] = np.sqrt((x_vec[i])**2 + (y_vec[i])**2 + (z_vec[i])**2)
#====================================================

#====================================================
# Per calcolare la retta congiungente i due punti 
x_values = [x_s[0], x_stop]
y_values = [y_s[0], y_stop]
z_values = [z_s[0], z_stop]
#====================================================    

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x_vec, y_vec, z_vec, c=r_vettore, cmap='cool', marker='o', linewidth=0.2)
ax.scatter(x_s, y_s, z_s, color='red', marker='o', linewidth=5)
ax.scatter(x_stop, y_stop, z_stop, color='red', marker='o', linewidths=5)
# Abbellimenti 
ax.scatter([], [], [], color='', label=r'$d_{tot}$ = %.2f cm'  %d_tot)
ax.scatter([], [], [], color='', label=r'$t_{volo}$ = %.2e s' %t_volo_tot)
ax.scatter([], [], [], color='', label=r'$N_{coll}$ = %.i'    %N_coll_tot)
ax.scatter([], [], [], color='', label=r'$r$     = %.2f cm'   %r)
ax.set_title(r'Traiettoria spaziale 3D - $^{12}C$ ')
ax.set_xlabel(r'x [cm]')
ax.set_ylabel(r'y [cm]')
ax.set_zlabel(r'z [cm]')
#ax.legend(loc='upper left')
fig.colorbar(p, ax=ax, label=r'r = $\sqrt{x_i^2 + y_i^2 + z_i^2}$')

# Retta tra i due punti 
ax.plot3D(x_values, y_values, z_values, color='orange')

ax.text(x_stop - 60, y_stop-20, z_stop,"Nucleo assorbitore\n($x_k$, $y_k$, $z_k$)", color='black')
ax.text(x_s[0], y_s[0], z_s[0]+10, "Sorgente", color='black')
plt.savefig('figures/neutron_path_simulation.png', format='png',bbox_inches="tight", dpi=100) 
plt.show()


