"""Compute the occurence of a given list of value. Plot a bar histrogram in a correct (I hope) way."""
import seaborn as sns
sns.set_theme(style="darkgrid")

import numpy as np
import matplotlib.pyplot as plt

# Dati inventati voti d'esame
voti_maschi  = [18, 18, 20, 20, 24, 30, 30, 24, 30, 22, 21, 21, 21, 24, 28, 28, 19, 18, 20]
voti_femmine = [18, 19, 22, 22, 25, 26, 30, 20, 30, 20, 19, 29, 29, 28, 28, 28, 19, 18, 20]

# Maschi
conteggi_m, bin_bordi_m = np.histogram(voti_maschi, bins=np.arange(min(voti_maschi), max(voti_maschi)+2))
print(f'conteggi: {conteggi_m}, len(conteggi): {len(conteggi_m)}')
print(f'bin_bordi: {bin_bordi_m}, len(bin_bordi): {len(bin_bordi_m)}')

#bin_centri  = 0.5 * (bin_bordi[1:] + bin_bordi[:-1])
bin_centri_m = bin_bordi_m[:-1]  # Escludi ultimo valore
print(f'bin_centri: {bin_centri_m}, len(bin_centri): {len(bin_centri_m)}')

plt.figure(1, figsize=(10, 5))
plt.xticks(np.arange(min(voti_maschi), max(voti_maschi)+1))


# Maschi
conteggi_f, bin_bordi_f = np.histogram(voti_femmine, bins=np.arange(min(voti_femmine), max(voti_femmine)+2))
print(f'conteggi: {conteggi_f}, len(conteggi): {len(conteggi_f)}')
print(f'bin_bordi: {bin_bordi_f}, len(bin_bordi): {len(bin_bordi_f)}')

#bin_centri  = 0.5 * (bin_bordi[1:] + bin_bordi[:-1])
bin_centri_f = bin_bordi_f[:-1]  # Escludi ultimo valore
print(f'bin_centri: {bin_centri_f}, len(bin_centri): {len(bin_centri_f)}')

plt.xticks(np.arange(min(voti_femmine), max(voti_femmine)+1))


plt.bar(bin_centri_m-0.1, conteggi_m, width=0.2, align='center',label="maschi")
plt.bar(bin_centri_f+0.1, conteggi_f, width=0.2, align='center', label="femmine")

plt.legend()
plt.xlabel("Voto esame")
plt.ylabel("Occorrenze")
plt.title("Voti esame: maschi vs femmine")

plt.savefig("figures/voti_esame.png")
plt.show()


