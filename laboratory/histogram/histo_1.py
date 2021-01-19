#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

# Data generatio
data = np.random.normal(0,1,1000)

# Make histogram
bin_heights, bin_borders, _ = plt.hist(data, bins=50, density=False, histtype='step')

# Plot
fig = plt.figure(1, figsize=(10,6))
plt.xlabel("variables")
plt.ylabel("# counts")
plt.minorticks_on()
plt.savefig('figures/easy_gauss_hist1.png', format='png',bbox_inches="tight", dpi=100)
