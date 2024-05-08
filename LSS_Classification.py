# Description: This script classifies the structures based on T-web classification for the given tidal shear tensor(s).
# The tidal shear tensor files are loaded and the eigenvalues and eigenvectors are calculated.
# The structures are classified based on T-web classification and the classification matrices are saved.

#-------------------------------------------------------------------------------------------------------------#
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from numba import jit
import os
from LSS_BB import *
#-------------------------------------------------------------------------------------------------------------#

# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, yn_smooth, \
        smoothing_scales, yn_potential, yn_traceless = read_input_file('input_params.txt')

# Smoothing the density field
if yn_smooth == 'y':

    def extract_scales(input_scales):

        scales = sorted(input_scales.split())

        truncated_scales = []
        sm_scales = []
        
        for scale in scales:
            sm_scales.append(float(scale))
            str_split = scale.split('.')
            truncated_scales.append(''.join(str_num for str_num in str_split))

        return sm_scales, truncated_scales
    
    smth_scales = extract_scales(smoothing_scales)[0]
    truncated_scales = extract_scales(smoothing_scales)[1]

tidal_path = os.path.join(save_path, 'tidal_fields')


if not os.path.exists(tidal_path):
    raise ValueError('The tidal tensor files do not exist. Please run the LSS_Tidal_shear.py script first.'
                        'Exiting the program....')


# Create a list of all the tidal tensor files (tidal_tensor_*.npy)
tidal_files = sorted([name for name in os.listdir(tidal_path) if name.startswith('tidal_tensor_')])
total_files = len(tidal_files)

print('Number of tidal field files found:', total_files)

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

for i, tidal_shear in tqdm(enumerate(tidal_shears)):

    if __name__ == '__main__':
        eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(tidal_shear_tensor=tidal_shear, grid_size=grid_size)

    save_path_eigenvalues = os.path.join(new_directory_path, f'evals_tidal_{truncated_scales[i]}.npy')
    save_path_eigenvectors = os.path.join(new_directory_path, f'evects_tidal_{truncated_scales[i]}.npy')

    print(f"{tidal_files[tidal_shears.index(tidal_shear)]} file has eigenvalues and eigenvectors calculated.\n")

    print('Saving the eigenvalues and eigenvectors....')
    save_data(eigenvalues, save_path_eigenvalues)
    save_data(eigenvalues, save_path_eigenvectors)

    # Append the eigenvalues and eigenvectors to the list
    tidal_eigenvalues.append(eigenvalues)
    tidal_eigenvectors.append(eigenvectors)

    print('Eigenvalues and eigenvectors saved successfully.\n'
          f'File name with suffix: *_{truncated_scales[i]}.npy indicating the corresponding smoothing scale.\n')

print('All eigenvalues and eigenvectors saved successfully.\n')

print('Classifying the structures based on T-web classification for the given tidal shear tensor(s)...')


new_directory_name = 'Classification matrices'
new_directory_path = os.path.join(save_path, new_directory_name)
create_directory(new_directory_path)

# Classify the structures based on T-web classification
for idx, num_file in enumerate(total_files):
    classification_matrix = np.zeros((grid_size, grid_size, grid_size))

    for i in tqdm(range(grid_size)):
        for j in range(grid_size):
            for k in range(grid_size):
                classification_matrix[i, j, k] = classify_structure(tidal_eigenvalues[num_file][i, j, k])

    
    save_path_classification = os.path.join(new_directory_path, f'classification_matrix_{truncated_scales[idx]}.npy')

    print('Saving the classification matrix....')
    save_data(classification_matrix, save_path_classification)
    print('Classification matrix saved successfully.\n'
          f'File name with suffix: *_{truncated_scales[idx]}.npy indicating the corresponding smoothing scale.\n')
    

print('All the structures are classified successfully.\nExiting the program....')



