"""Creation of a table (DataFrame) of the loaded data."""

import numpy as np
import pandas as pd

'''
The Pandas library is able of creating tables and much more
readable, and even to translate very quickly into code for a LaTeX document.
'''

# Load data file
x, y, dy = np.loadtxt("../data/data_1.txt", comments='#', unpack=True)

# Creating a dictionary containing the data (required for Pandas)
data = {'x [u.m.]': x, 'y [u.m.]': y, 'dy [u.m.]': dy}

# Creating a dataframe containing the data
table = pd.DataFrame(data)
print(table)

# To approximate values ​​to 2 decimal places, use the method: '.round(2)'
table = pd.DataFrame(data).round(2)
print('\n---------------------------------\n')
print(table)


# Save table in *.csv format
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
table.to_csv('tables/data_1.csv', index=False)