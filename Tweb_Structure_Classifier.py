#------------------------------------------------------------------------------------------------------------------------------#
# Description: This script classifies the structures based on T-web classification for the given tidal shear tensor(s).        #
#              The tidal shear tensor files are loaded and the eigenvalues and eigenvectors are calculated for respective      #   
#              smoothing scales. The structures are classified based on T-web classification and the classification matrices   #     
#------------------------------------------------------------------------------------------------------------------------------#
 
#### INSTRUCTIONS ####
# 1. Make sure the input_params.txt file is present in the current working directory.
# 2. Make sure the Tidal_Field_Calculator.py script is run before running this script.

# Import necessary libraries
from tqdm import tqdm
import os
from LSS_TWeb_BlackBox import *
import time

# Start the timer
start_time = time.time()

#----------------------------------------- READ INPUT FILE ----------------------------------------------#

# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, \
        smoothing_scales, yn_potential, yn_traceless = read_input_file('input_params.txt')

#------------------------------ EXTRACT SMOOTHING SCALES --------------------------------#

def extract_smoothing_scales(smoothing_scales: list[float]) -> tuple[list[float], list[str]]:
    """
    Extract and truncate the smoothing scales.

    Parameters:
    - smoothing_scales (list[float]): List of smoothing scales.

    Returns:
    - tuple: A tuple containing two lists:
        - list[float]: The original floating-point smoothing scales.
        - list[str]: The truncated string forms of the smoothing scales, with the decimal point removed.

    """  
    # Extract the smoothing scales
    smth_scales = extract_scales(smoothing_scales)[0] # Extract the smoothing scales (float)
    truncated_scales = extract_scales(smoothing_scales)[1] # Truncate the scales to remove the decimal point

    return smth_scales, truncated_scales

#------------------------------ LOAD THE TIDAL SHEAR TENSOR FILES --------------------------------#

def load_tidal_shear_files() -> None:
    """
    Load the tidal shear files from the specified directory.

    Raises:
    - ValueError: If the tidal tensor files do not exist.

    Returns:
    - tidal_shears: A list of loaded tidal shear files.
    """
    tidal_path = os.path.join(save_path, 'tidal_fields')

    if not os.path.exists(tidal_path):
        raise ValueError('The tidal tensor files do not exist. Please run the LSS_Tidal_shear.py script first.'
                            'Exiting the program....')
        exit()

    print('Loading the tidal tensor files....')

    # Load the tidal tensor files
    tidal_shears = load_all_npy_files(folder_path=tidal_path, filename_prefix='tidal_tensor_')

    print('Tidal tensor files loaded successfully.\n')

    total_files = len(tidal_shears)

    print('Number of tidal field files found:', total_files, '\n')

    return tidal_shears


#------------------------------ CALCULATE EIGENVALUES AND EIGENVECTORS OF TIDAL SHEAR TENSOR --------------------------------#

def calculate_tidal_eigenvalues_and_eigenvectors(tidal_shears: list[np.ndarray], truncated_scales: list[str]) -> None:
    """
    Calculate and save the eigenvalues and eigenvectors of the tidal shear tensor for each smoothing scale.

    Parameters:
    - tidal_shears (list of numpy.ndarray): List of tidal shear tensors for different smoothing scales.
    - truncated_scales (list of str): List of truncated smoothing scale strings used for naming the output files.

    Returns:
    - all_tidal_eigenvalues (dict): Dictionary containing eigenvalues for each smoothing scale.
    - all_tidal_eigenvectors (dict): Dictionary containing eigenvectors for each smoothing scale.
    """
    # Create a directory to save the eigenvalues and eigenvectors
    new_directory_name = 'Tidal eigenvalues and eigenvectors'
    new_directory_path = os.path.join(save_path, new_directory_name)
    create_directory(new_directory_path)

    print('\nCalculating the eigenvalues and eigenvectors of the tidal shear tensor....')

    # Initialize the lists to store the eigenvalues and eigenvectors
    all_tidal_eigenvalues = {}
    all_tidal_eigenvectors = {}

    for i in tqdm(range(len(tidal_shears))):

        eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(tidal_shear_tensor=tidal_shears[i], grid_size=grid_size)

        save_path_eigenvalues = os.path.join(new_directory_path, f'evals_tidal_{truncated_scales[i]}.npy')
        save_path_eigenvectors = os.path.join(new_directory_path, f'evects_tidal_{truncated_scales[i]}.npy')

        print('Saving the eigenvalues and eigenvectors....')
        save_data(data=eigenvalues, file_path=save_path_eigenvalues)
        save_data(data=eigenvectors, file_path=save_path_eigenvectors)

        # Append the eigenvalues and eigenvectors to the dictionary
        all_tidal_eigenvalues[truncated_scales[i]] = eigenvalues
        all_tidal_eigenvectors[truncated_scales[i]] = eigenvectors


    print(f'\nEigenvectors has shape: {eigenvectors.shape}\n'
            f'Eigenvalues has shape: {eigenvalues.shape}\n')

    print('\nEigenvalues and eigenvectors saved successfully.\n'
            f'File name with suffix: *_{truncated_scales[i]}.npy indicating the corresponding smoothing scale.\n')
    
    return all_tidal_eigenvalues, all_tidal_eigenvectors


#------------------------------ T WEB CLASSIFICATION --------------------------------#

def classify_structures(all_tidal_eigenvalues: list[np.ndarray]) -> None:
    """
    Classify structures based on the T-web classification scheme using the tidal shear tensor eigenvalues.

    This function classifies the cosmic web structures into voids, sheets, filaments, and knots 
    based on the eigenvalues of the tidal shear tensor. It saves the classification matrices for each 
    smoothing scale in a specified directory.

    Parameters:
    - all_tidal_eigenvalues (list of numpy.ndarray): A list of numpy arrays containing the eigenvalues 
      of the tidal shear tensor for different smoothing scales.

    Returns:
    - None
    """
    
    print('Classifying the structures based on T-web classification for the given tidal shear tensor(s)...')

    # Create a directory to save the classification matrices
    new_directory_path = os.path.join(save_path, 'Classification matrices')
    create_directory(new_directory_path)

    # Classify the structures based on T-web classification
    for i in range(len(truncated_scales)):

        classification_matrix = classify_structure(eigenvalues=all_tidal_eigenvalues[truncated_scales[i]])
        
        save_path_classification = os.path.join(new_directory_path, f'classification_matrix_{truncated_scales[i]}.npy')

        print('Saving the classification matrix....')
        save_data(data=classification_matrix, file_path=save_path_classification)
        print('Classification matrix saved successfully.\n'
            f'File name with suffix: *_{truncated_scales[i]}.npy indicating the corresponding smoothing scale.\n')
    
    return None
       

if __name__ == '__main__':
    
    smth_scales, truncated_scales = extract_smoothing_scales(smoothing_scales)

    tidal_shears = load_tidal_shear_files()

    all_tidal_eigenvalues, all_tidal_eigenvectors = calculate_tidal_eigenvalues_and_eigenvectors(tidal_shears=tidal_shears, truncated_scales=truncated_scales)

    classify_structures(all_tidal_eigenvalues=all_tidal_eigenvalues)



# Stop the timer
end_time = time.time()
print('Time taken to run the complete script:', end_time - start_time)

print('All the structures are classified successfully.\nExiting the program....')



#----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#