# https://seaborn.pydata.org/generated/seaborn.pairplot.html

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


"""The simplest invocation uses scatterplot() for each pairing 
of the variables and histplot() for the marginal plots along the diagonal: """


# Loading data from seaborn
penguins = sns.load_dataset("penguins")

print(penguins)

# Make pariplot
# Assigning a hue variable adds a semantic mapping and changes 
# the default marginal plot to a layered kernel density estimate (KDE):
sns.pairplot(penguins, hue="species")

plt.savefig("figures/pairplot_penguins.png")
plt.show()