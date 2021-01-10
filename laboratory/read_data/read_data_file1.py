"""Script to read data file."""
#==============================
# Mode 1
#==============================
# (more convenient if the arrangement of the data in the file is unknown)

#file_data = open("file_data.txt", "r")
file_data = open("data/data.txt")
data = file_data.read()
print(data)


#==============================
# Mode 2 
#==============================
import numpy as np

# Load data file
data = np.loadtxt("data/data.txt", comments='#', unpack=True)
print('data:', data)

# Print x, y, z
x = data[0,:]
y = data[1,:]
dy = data[2,:]
print('\n--------------------------------\n')
print('x  = ',x)
print('y  = ',y)
print('dy = ',dy)


#==============================
# Mode 3 (faster)
#==============================
x, y, dy = np.loadtxt("data/data.txt", comments='#', unpack=True)

# Print x, y, z
print('x  = ',x)
print('y  = ',y)
print('dy = ',dy)