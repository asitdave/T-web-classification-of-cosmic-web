#----------------------------------------------------------------------------------------------------------------------#
# Description: This script analyses the classification of structures based on the T-web classification. The volume fractions
#              of different structures are plotted against the smoothing scales. The classification overlay on the density
#              field is plotted for the given smoothing scales. The changes in the classification of particles are also
#              plotted if the smoothing scales are changed.
#----------------------------------------------------------------------------------------------------------------------#
 
#### INSTRUCTIONS ####
# 1. Make sure the input_params.txt file is present in the current working directory.
# 2. Make sure the LSS_Tidal_shear.py & LSS_Classification.py script is ran before running this script.
# 3. Parameters that can be modified in this script are:
#       - Slice index for the structure classification (default: middle slice)
#       - Projection of the box for the plots (default: xy)
#       - Thickness of the slice (to average the density field over) for the classification 
#         overlay plots (default: half box thickness)


import numpy as np
import os
from LSS_TWeb_BlackBox import *
import time

# Start the timer
start_time = time.time()

#----------------------------------------- READ INPUT FILE ----------------------------------------------#
# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, \
        smoothing_scales, calculate_potential, calculate_traceless = read_input_file('input_params.txt')

def get_box_size():
    """
    Extract the box size from the 'simulation_properties.txt' file.

    This function reads the 'simulation_properties.txt' file to find the line
    that contains the 'Box size' information, extracts the numerical value of
    the box size, and returns it as an integer.

    Returns:
    - int: The extracted box size.

    Raises:
    - FileNotFoundError: If 'simulation_properties.txt' does not exist.
    - ValueError: If the box size cannot be found or parsed correctly.

    """
    try:
        # Open the file and read lines
        with open('simulation_properties.txt', "r") as f:
            lines = f.readlines()
            
        # Search for the line that starts with 'Box size' and extract the value
        for line in lines:
            if line.startswith('Box size'):
                box_size = int(line.split(':')[1].strip().split(' ')[0])
                return box_size
        
        # If 'Box size' is not found, raise an error
        raise ValueError("Box size not found in the 'simulation_properties.txt' file.")
    
    except FileNotFoundError:
        raise FileNotFoundError("'simulation_properties.txt' file does not exist.")
    
    except ValueError as e:
        raise ValueError(f"Error in parsing box size: {e}")

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

#------------------------------ LOAD THE DENSITY FIELD FILES --------------------------------#

def load_density_fields() -> np.ndarray:
    """
    Load smoothed density fields from the specified directory.

    This function retrieves all numpy files from the directory specified by 
    'save_path/smoothed_density_fields' that have filenames starting with 
    'smoothed_density_field_' and returns them as a numpy array.

    Returns:
    - np.ndarray: An array containing all loaded smoothed density fields.
    """
    # Load smoothed density field
    density_fields = load_all_npy_files(folder_path=os.path.join(save_path, 'smoothed_density_fields'), filename_prefix='smoothed_density_field_')
    return density_fields

#------------------------------ LOAD THE CLASSIFICATION MATRICES --------------------------------#

def load_classification_matrices() -> np.ndarray:
    """
    Load classification matrices from the specified directory.

    This function retrieves all numpy files from the directory specified by 
    'save_path/Classification matrices' that have filenames starting with 
    'classification_matrix_' and returns them as a numpy array.

    Returns:
    - np.ndarray: An array containing all loaded classification matrices.
    """
    # Create path to save the classification matrices
    classification_path = os.path.join(save_path, 'Classification matrices')

    print('Loading the classification matrix files....')

    # Load the classification matrix files
    classifications = load_all_npy_files(folder_path=classification_path, filename_prefix='classification_matrix_') # for all smoothing scales
    total_files = len(classifications)


    print('Number of files found:', total_files)
    print('Classification matrix files loaded successfully.\n')

    return classifications

#------------------------------ CREATE A DIRECTORY FOR PLOTS --------------------------------#

def create_directory_for_plots():
    """
    Create a directory to save the classification analysis plots.

    This function creates a new directory named 'Classification analysis plots' 
    within the specified save path. If the directory already exists, it will not 
    create a new one but return the path to the existing directory.

    Returns:
    - str: The path to the created or existing directory for classification analysis plots.
    """
    # Create a directory to save the classification analysis plots
    new_directory_name = 'Classification analysis plots'
    new_directory_path = os.path.join(save_path, new_directory_name)
    create_directory(new_directory_path)

    return new_directory_path

#------------------------------ PLOT VOLUME FRACTIONS VS RS --------------------------------#

def plot_volfrac_vs_rs(classifications: np.ndarray, smooth_scales: list[float], dir_path: str):
    """
    Plot the volume fractions vs Rs.

    This function generates a plot of volume fractions against the smoothing scales (Rs) 
    for the given classification matrices. It saves the plot as 'Volume_fractions_vs_Rs.png' 
    in the specified directory path.

    Parameters:
    - classifications (np.ndarray): An array containing the classification matrices.
    - smooth_scales (list[float]): List of smoothing scales.
    - dir_path (str): The directory path to save the plot.

    Returns:
    - None
    """
    # Plot the volume fractions vs Rs
    plot_volfrac_rs(classifications = classifications, 
                    smooth_scales = smooth_scales, 
                    save_path = os.path.join(dir_path, 'Volume_fractions_vs_Rs.png'))

#------------------------------ PLOT CLASSIFICATION OVERLAY ON DENSITY FIELD --------------------------------#

def plot_structure_classification(classifications: np.ndarray, density_fields: np.ndarray, smooth_scales: list[float], 
                                  truncated_scales: list[str], dir_path: str, slice_thickness: list[int, int], slice_index: int, projection: str):
    """
    Plot the structure classification overlay on the density field for all smoothing scales.

    This function plots the classification overlay on the density field for each smoothing scale. 
    It saves the plots with filenames indicating the corresponding smoothing scales in the specified directory path.

    Parameters:
    - classifications (np.ndarray): An array containing the classification matrices for all smoothing scales.
    - density_fields (np.ndarray): An array containing the density fields for all smoothing scales.
    - smooth_scales (list[float]): List of smoothing scales.
    - truncated_scales (list[str]): List of truncated smoothing scale names.
    - dir_path (str): The directory path to save the plots.
    - slice_thickness (list[int, int]): The thickness of the slice for the classification overlay plots. [start_index, end_index]
    - slice_index (int): The index of the slice for the classification overlay plots. 
    - projection (str): The projection of the box for the classification overlay plots.

    Returns:
    - None
    """
    # Plot the classification overlay on density field for all smoothing scales
    for i in range(len(smooth_scales)):
        plot_classification_overlay(smth_scale = smooth_scales[i], 
                                    classification = classifications[i], 
                                    rho = density_fields[i], 
                                    slice_thickness = slice_thickness, 
                                    slice_index = slice_index, 
                                    projection = projection, 
                                    grid_size=grid_size, box_size=box_size, 
                                    save_path = os.path.join(dir_path, f'Classification_overlay_{truncated_scales[i]}.png'))

#------------------------------ GET STRUCTURE CHANGES --------------------------------#

def get_structure_changes(classifications: np.ndarray, slice_index: int, projection: str) -> dict[str, np.ndarray]:
    """
    Get structure changes between two classification matrices.

    This function calculates the structure changes between two classification matrices, typically representing the beginning 
    and end of a simulation. It returns a dictionary containing the changes in the cosmic web structures.

    Parameters:
    - classifications (np.ndarray): An array containing the classification matrices for different time steps.
    - slice_index (int): The index of the slice to analyze for structure changes.
    - projection (str): The projection type ('xy', 'yz', 'xz') to analyze for structure changes.

    Returns:
    - dict: A dictionary containing the changes in the cosmic web structures.
    """
    # Get all environment changes
    env_changes = get_env_changes(clf1=classifications[0], clf2=classifications[-1], 
                                slice_index=slice_index, 
                                projection=projection, 
                                grid_size=grid_size, box_size=box_size)
    
    return env_changes

#------------------------------ GENERATE TRANSFORMATION DICTIONARY --------------------------------#

def generate_transformation_dictionary(env_changes: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Generate two transformation dictionaries based on environmental changes.

    Parameters:
    - env_changes (dict): A dictionary containing environmental changes.

    Returns:
    - tuple: A tuple containing two dictionaries representing transformations.
             Each dictionary has string keys indicating the type of transformation
             and NumPy arrays as values representing the changed environments.
    """
    # Generate the transformation dictionary
    transformations_1 = {
        'Void to Sheet': env_changes['Void to Sheet'],
        'Sheet to Filament': env_changes['Sheet to Filament'],
        'Filament to Node': env_changes['Filament to Node'],
        'Sheet to Void': env_changes['Sheet to Void'],
        'Filament to Sheet': env_changes['Filament to Sheet'],
        'Node to Filament': env_changes['Node to Filament']
    }

    # Plot the second set of transformations
    transformations_2 = {
        'Void to Filament': env_changes['Void to Filament'],
        'Void to Node': env_changes['Void to Node'],
        'Sheet to Node': env_changes['Sheet to Node'],
        'Filament to Void': env_changes['Filament to Void'],
        'Node to Void': env_changes['Node to Void'],
        'Node to Sheet': env_changes['Node to Sheet']
    }

    return transformations_1, transformations_2

#------------------------------ PLOT STRUCTURE CHANGES --------------------------------#

def plot_structure_changes(transformations: dict[str, np.ndarray], density_fields: np.ndarray, truncated_scales: list[str],  dir_path: str, 
                           slice_thickness: list[int, int], slice_index: int, projection: str, unique_num: int) -> None:
    """
    Plot the changes in classification of halo environments for given transformations.

    Parameters:
    - transformations (dict): A dictionary containing environmental changes.
    - density_fields (np.ndarray): An array containing density fields.
    - truncated_scales (list[str]): A list of truncated smoothing scale names.
    - dir_path (str): The directory path to save the plots.
    - slice_thickness (list[int, int]): The range of slices to average over [Start index, End index].
    - slice_index (int): The index of the slice to plot.
    - projection (str): The projection type ('xy', 'yz', 'xz').
    - unique_num (int): An unique identifier to differentiate between plots.

    Returns:
    - None
    """
    # Plot the changes in classification of Halo environments for above transformations
    plot_env_changes(transformations = transformations, 
                    density_slice = slice_density_field(density_fields[slice_index], 
                                                        slice_thickness=slice_thickness,
                                                        projection=projection 
                                                        ),
                    box_size=box_size, 
                    title = 'Changes in classification of Halo environments', 
                    save_path = os.path.join(dir_path, f'Classification_change_{truncated_scales[0]}_{truncated_scales[-1]}_{unique_num}.png'))




if __name__ == '__main__':

    # Get the box size
    box_size = get_box_size()
    
    # Extract the smoothing scales
    smooth_scales, truncated_scales = extract_smoothing_scales(smoothing_scales)

    # Load the density fields that were saved using LSS_Tidal_shear.py
    density_fields = load_density_fields()

    # Load the classification matrices that were saved using LSS_Classification.py
    classifications = load_classification_matrices()

    # Create a directory to save the classification analysis plots
    new_directory_path = create_directory_for_plots()

    # Plot the classification overlay on density field for all smoothing scales
    plot_structure_classification(classifications, density_fields, smooth_scales, truncated_scales, new_directory_path,
                                  slice_thickness=[grid_size//2, grid_size//2 + 1], # Specify the range of slices to average the density field over [Start index, End index]
                                  slice_index = len(density_fields)//2, # Specify the slice index you want to consider for structure classification
                                  projection = 'xy' # Specify the projection of the box for classification overlay
                                  )

    if len(smooth_scales) > 1:
        # Plot the volume fractions vs Rs
        plot_volfrac_vs_rs(classifications, smooth_scales, new_directory_path)

        # Get all environment changes
        env_changes = get_structure_changes(classifications, 
                                            slice_index=grid_size//2, # Specify the slice index you want to consider for structure classification
                                            projection='xy' # Specify the projection of the box for classification overlay
                                            )

        # Generate the transformation dictionary
        transformations_1, transformations_2 = generate_transformation_dictionary(env_changes)

        # Plot the changes in classification of Halo environments for above transformations
        plot_structure_changes(transformations_1, density_fields, truncated_scales, new_directory_path, 
                            slice_thickness=[grid_size//2, grid_size//2 + 1], # Specify the range of slices to average the density field over [Start index, End index]
                            slice_index = len(density_fields)//2, # Specify the slice index you want to consider for structure classification
                            projection = 'xy', # Specify the projection of the box for classification overlay
                            unique_num=1
                            )
        
        plot_structure_changes(transformations_2, density_fields, truncated_scales, new_directory_path, 
                            slice_thickness=[grid_size//2, grid_size//2 + 1], # Specify the range of slices to average the density field over [Start index, End index]
                            slice_index = len(density_fields)//2, # Specify the slice index you want to consider for structure classification
                            projection = 'xy', # Specify the projection of the box for classification overlay
                            unique_num=2
                            )

    else: 
        print('Volume fraction plot requires more than one smoothing scale to compare.')
        print('Skipping the volume fraction plot & changes in structure classification plots....\n')


# Stop the timer
end_time = time.time()
print('Time taken to run the complete script:', end_time - start_time)

#----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#