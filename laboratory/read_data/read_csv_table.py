"""
Script implementing a function to open .csv file and print the file on terminal.
"""

import numpy as np
import pandas as pd


def read_csv(file_path):
    """Function to open and storage data containted in a .cvs table.
    """
    dataframe = pd.read_csv(file_path)
    
    return dataframe
    
    
if __name__ == "__main__":
    # Load data
    data = read_csv('data/binding_energy.csv')
    # https://www.roma1.infn.it/people/dionisi/triennale/cap11-fisica-nucleare.pdf
    # https://www-nds.iaea.org/radii/
    print(data)
    
    # I can work on the data in the same way
    # that I work with the pandas dataframe
    print(data['Z'])

    # How to assign values ​​to individual variables
    Z                  = data['Z'].to_numpy(dtype=np.float32)
    N                  = data['N'].to_numpy(dtype=np.float32)
    A                  = data['A'].to_numpy(dtype=np.float32)
    elements           = data['nucleus'].to_numpy(dtype=str)
    B_over_A_exp       = data['B_over_A_exp'].to_numpy(dtype=np.float32)
    
    #print(B_over_A_exp)
    
