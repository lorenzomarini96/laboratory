import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# for LaTeX
from matplotlib import rcParams
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

# Lista dei voti d'esame
voti = ["18", "19", "20", "21", "22",
        "23", "24", "25", "26", "27",
        "28", "29", "30"]

# Maschi
numero_maschi = [10, 2, 4, 10, 1,
                 0, 0, 10, 11, 12,
                 1, 1, 4]

# Femmine
numero_femmine = [6, 0, 0, 1, 12,
                 2, 3, 4, 4, 2,
                 1, 5, 6]

# X axis
x = np.arange(len(voti))
width = 0.35 # Larghezza delle barre

fig = plt.figure(1, figsize=(10,6))
ax  = plt.axes()
maschi = ax.bar(x - width/2, numero_maschi, width, label="Maschi")
femmine = ax.bar(x + width/2, numero_femmine, width, label="Femmine")

# Add some text for labels and others...
ax.set_ylabel("Voti esame")
ax.set_title("Istogramma a barre dei voti di esame")
ax.set_xticks(x)
ax.set_xticklabels(voti)
ax.legend()


# autolabel
def autolabel(studenti):
    """Appende un testo sopra ciascuna barra, mostrando il
    valore del voto corrispondente."""
    for studente in studenti:
        altezza = studente.get_height()
        ax.annotate('{}'.format(altezza),
        xy=(studente.get_x() + studente.get_width()/2 ,altezza),
        xytext=(0, 3), # 3 point verical offset
        textcoords = "offset points",
        ha = "center", va = "bottom")

autolabel(maschi)
autolabel(femmine)  

fig.tight_layout()
#plt.savefig('figures/bar_histo.png', bbox_inches='tight')
plt.show()