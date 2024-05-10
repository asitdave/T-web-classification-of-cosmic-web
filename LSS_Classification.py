#----------------------------------------------------------------------------------------------------------------------#
# Description: This script classifies the structures based on T-web classification for the given tidal shear tensor(s).
# The tidal shear tensor files are loaded and the eigenvalues and eigenvectors are calculated.
# The structures are classified based on T-web classification and the classification matrices are saved.
#----------------------------------------------------------------------------------------------------------------------#
 
# Import necessary libraries
import numpy as np
from tqdm import tqdm
import os
from LSS_BB import *
import time

# Start the timer
start_time = time.time()

#----------------------------------------- READ INPUT FILE ----------------------------------------------#

# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, yn_smooth, \
        smoothing_scales, yn_potential, yn_traceless = read_input_file('input_params.txt')

# Smoothing the density field
if yn_smooth == 'y':
    
    smth_scales = extract_scales(smoothing_scales)[0]
    truncated_scales = extract_scales(smoothing_scales)[1]

# Locate the tidal tensor files
tidal_path = os.path.join(save_path, 'tidal_fields')

if not os.path.exists(tidal_path):
    raise ValueError('The tidal tensor files do not exist. Please run the LSS_Tidal_shear.py script first.'
                        'Exiting the program....')
    exit()

#------------------------------ LOAD THE TIDAL SHEAR TENSOR FILES --------------------------------#

# Create a list of all the tidal tensor files (tidal_tensor_*.npy)
tidal_files = sorted([name for name in os.listdir(tidal_path) if name.startswith('tidal_tensor_')])
total_files = len(tidal_files)

print('Number of tidal field files found:', total_files)

print('Loading the tidal tensor files....')

# Load the tidal tensor files
tidal_shears = load_all_npy_files(tidal_path)

print('Tidal tensor files loaded successfully.\n')


#------------------------------ CALCULATE EIGENVALUES AND EIGENVECTORS OF TIDAL SHEAR TENSOR --------------------------------#

new_directory_name = 'Tidal eigenvalues and eigenvectors'
new_directory_path = os.path.join(save_path, new_directory_name)
create_directory(new_directory_path)

print('Calculating the eigenvalues and eigenvectors of the tidal shear tensor....')

all_tidal_eigenvalues = []
all_tidal_eigenvectors = []

for tidal_shear in tqdm(tidal_shears):

    eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(tidal_shear_tensor=tidal_shear, grid_size=grid_size)

    save_path_eigenvalues = os.path.join(new_directory_path, f'evals_tidal_{truncated_scales[tidal_shears.index(tidal_shear)]}.npy')
    save_path_eigenvectors = os.path.join(new_directory_path, f'evects_tidal_{truncated_scales[tidal_shears.index(tidal_shear)]}.npy')

    print('Saving the eigenvalues and eigenvectors....')
    save_data(eigenvalues, save_path_eigenvalues)
    save_data(eigenvalues, save_path_eigenvectors)

    # Append the eigenvalues and eigenvectors to the list
    all_tidal_eigenvalues.append(eigenvalues)
    all_tidal_eigenvectors.append(eigenvectors)


print(f'\nEigenvectors has shape: {eigenvectors.shape}\n'
        f'Eigenvalues has shape: {eigenvalues.shape}\n')

print('\nEigenvalues and eigenvectors saved successfully.\n'
        f'File name with suffix: *_{truncated_scales[tidal_shears.index(tidal_shear)]}.npy indicating the corresponding smoothing scale.\n')


#------------------------------ T WEB CLASSIFICATION --------------------------------#

print('Classifying the structures based on T-web classification for the given tidal shear tensor(s)...')

new_directory_name = 'Classification matrices'
new_directory_path = os.path.join(save_path, new_directory_name)
create_directory(new_directory_path)

# Classify the structures based on T-web classification
for tidal_eigenvalues in all_tidal_eigenvalues:
    classification_matrix = classify_structure(tidal_eigenvalues)
    
    save_path_classification = os.path.join(new_directory_path, f'classification_matrix_{truncated_scales[tidal_eigenvalues.index(tidal_eigenvalues)]}.npy')

    print('Saving the classification matrix....')
    save_data(classification_matrix, save_path_classification)
    print('Classification matrix saved successfully.\n'
          f'File name with suffix: *_{truncated_scales[tidal_eigenvalues.index(tidal_eigenvalues)]}.npy indicating the corresponding smoothing scale.\n')
    

# Stop the timer
end_time = time.time()
print('Time taken to run the complete script:', end_time - start_time)

print('All the structures are classified successfully.\nExiting the program....')



#----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#