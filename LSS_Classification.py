import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import itertools
from numba import jit
import os
from LSS_BB import *
import time
import shutil

while True:
    try:
        tidal_path = str(input('Give the path where all tidal tensor files (.npy) are stored: '))
        if not os.path.exists(tidal_path):
            raise ValueError('Please enter a valid file path.')
        break
    except ValueError as e:
        print(e)

while True:
  try:
    grid_size = float(input('Enter the grid size: '))
    break  # Exit the loop if input is valid
  except ValueError:
    print('Please enter a valid grid size. (float)')

try:
    save_path = str(input('Give the path where you want to save the results: '))
    if not os.path.exists(save_path):
        raise ValueError('Please enter a valid file path.')
except ValueError as e:
    print(e)


tidal_files = [name for name in os.listdir(tidal_path) if name.startswith('tidal_tensor')]
total_files = len(tidal_files)

print('Number of files found:', total_files)

print('Loading the tidal tensor files....')

# Load the tidal tensor files
tidal_shears = load_all_npy_files(tidal_path)

print('Tidal tensor files loaded successfully.\n')

new_directory_name = 'Tidal eigenvalues and eigenvectors'
new_directory_path = os.path.join(save_path, new_directory_name)
create_directory(new_directory_path)


print('Calculating the eigenvalues and eigenvectors of the tidal shear tensor....')

# Calculate the eigenvalues and eigenvectors of the tidal shear tensor

tidal_eigenvalues = []
tidal_eigenvectors = []

for tidal_shear in tqdm(tidal_shears):
    eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(tidal_shear)

    save_path_eigenvalues = os.path.join(new_directory_path, f'evals_tidal_{tidal_shears.index(tidal_shear)}.npy')
    save_path_eigenvectors = os.path.join(new_directory_path, f'evects_tidal_{tidal_shears.index(tidal_shear)}.npy')

    print(f"{tidal_files[tidal_shears.index(tidal_shear)]} file has eigenvalues and eigenvectors calculated.\n")

    print('Saving the eigenvalues and eigenvectors....')
    save_data(eigenvalues, save_path_eigenvalues)
    save_data(eigenvalues, save_path_eigenvectors)

    # Append the eigenvalues and eigenvectors to the list
    tidal_eigenvalues.append(eigenvalues)
    tidal_eigenvectors.append(eigenvectors)

    print('Eigenvalues and eigenvectors saved successfully.\n'
          f'File name with suffix: *_{tidal_shears.index(tidal_shear)}.npy')

print('All eigenvalues and eigenvectors saved successfully.\n')

print('Classifying the structures based on T-web classification for the given tidal shear tensor(s)...')


new_directory_name = 'Classification matrices'
new_directory_path = os.path.join(save_path, new_directory_name)
create_directory(new_directory_path)

# Classify the structures based on T-web classification
for num_file in range(total_files):
    classification_matrix = np.zeros((grid_size, grid_size, grid_size))

    for i in tqdm(range(grid_size)):
        for j in range(grid_size):
            for k in range(grid_size):
                classification_matrix[i, j, k] = classify_structure(tidal_eigenvalues[num_file][i, j, k])

    
    save_path_classification = os.path.join(new_directory_path, f'classification_matrix_{num_file}.npy')

    print('Saving the classification matrix....')
    save_data(classification_matrix, save_path_classification)
    print('Classification matrix saved successfully.\n'
          f'File name with suffix: *_{num_file}.npy')
    

print('All the structures are classified successfully.\nExiting the program....')



