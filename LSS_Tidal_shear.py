#------------------------------------------------------------------------------------------------------------------------------#
# Description: This script computes the density field, tidal tensor, potential field, and traceless tidal shear tensor for a   #
#              given pynbody.snapshot. The user can also smooth the density field using a Gaussian filter and also choose if   #
#              they want to compute the potential field and the traceless tidal shear.                                         #     
#------------------------------------------------------------------------------------------------------------------------------#

#### INSTRUCTIONS ####
# 1. Make sure the input_params.txt file is present in the current working directory.
# 2. Make sure the LSS_BB.py script is present in the current working directory.
# 3. The script expects you to have a simulation snapshot for the analysis.
# 4. Parameters that can be modified in this script are:
#       - Projection of the box for the plots (default: xy)
#       - Thickness of the slice to average the density field over. (default: half box thickness)

# Importing the necessary libraries
from tqdm import tqdm
import os
from LSS_TWeb_BlackBox import *
import time

# Start the timer
start_time = time.time()

#------------------------------ READ INPUT FILE ----------------------------------#

# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, \
        smoothing_scales, calculate_potential, calculate_traceless = read_input_file('input_params.txt')

print('\nInput parameters read succefully. Please wait...\n')


#------------------------------ LOAD THE SNAPSHOT --------------------------------#

def get_snapshot(snapshot_path: str) -> tuple[pynbody.snapshot.SimSnap, pynbody.simdict.SimDict]:
    """
    Load the snapshot from the specified path.

    Parameters:
    - snapshot_path (str): The file path to the snapshot.

    Returns:
    - tuple: A tuple containing the loaded snapshot and its header.

    """
    print('Loading the snapshot...')
    snap, header = load_snapshot(snapshot_path)
    print('Snapshot loaded successfully.')

    return snap, header

#------------------------------ EXTRACT SIMULATION PROPERTIES --------------------------------#

def extract_simulation_params(snapshot_header: pynbody.simdict.SimDict) -> dict[str, float]:
    """
    Extract values from the simulation dictionary.

    Parameters:
    - snapshot_header (pynbody.simdict.SimDict): The simulation dictionary.

    Returns:
    - extracted_values: A dictionary containing the values ['omegaM0', 'omegaL0', 'a', 'h']
                        for corresponding simulation snapshot.

    """
    # Extract the simulation properties
    extracted_values = extract_simdict_values(simdict=snapshot_header)

    return extracted_values

#------------------------------ SAVE THE SIMULATION PROPERTIES --------------------------------#

def save_simulation_properties(snapshot_header: pynbody.simdict.SimDict, snapshot: pynbody.snapshot.SimSnap) -> None:
    """
    Save the simulation properties to a file.

    Parameters:
    - snapshot_header (pynbody.simdict.SimDict): The simulation header.
    - snapshot (pynbody.snapshot.SimSnap): The loaded snapshot.

    """
    # Save the simulation properties to a file
    with open('simulation_properties.txt', "w") as f:
        f.write('-------------------------------------------------------------\n')
        f.write(' The file contains the simulation properties and parameters \n')
        f.write('-------------------------------------------------------------\n\n')
        
        f.write('Box size: ' + str(snapshot_header['boxsize']) + '\n')
        f.write('Omega_M0: ' + str(snapshot_header['omegaM0']) + '\n')
        f.write('Omega_L0: ' + str(snapshot_header['omegaL0']) + '\n')
        f.write('a: ' + str(snapshot_header['a']) + '\n')
        f.write('h: ' + str(snapshot_header['h']) + '\n')
        f.write('time: ' + str(snapshot_header['time']) + '\n\n')
        
        mean_mass = snapshot['mass'].mean()
        f.write('Mass of each particle: ' + str(mean_mass) + ' * ' + str(mean_mass.units) + '\n')
        
        total_particles = snapshot['pos'].shape[0]
        f.write('Total number of particles: ' + str(total_particles) + '\n')

    print('Simulation properties saved successfully. Please check the "simulation_properties.txt" file in cwd.\n\n')

#------------------------------ SAVE THE PARTICLE POSITIONS AND VELOCITIES --------------------------------#

def save_particle_positions_and_velocities(snapshot: pynbody.snapshot.SimSnap) -> None:
    """
    Save the particle positions and velocities.

    Parameters:
    - snapshot (pynbody.snapshot.SimSnap): The loaded snapshot.

    """
    try:
        # Create path to save the particle positions
        save_path_particle_position = os.path.join(save_path, 'particle_positions.npy')

        # Save the particle positions
        save_data(data = snapshot['pos'], file_path=save_path_particle_position)
        print("Particle position file ('particle_positions.npy') saved successfully.")

    except:
        print('Error in saving the particle positions. Skipping the step....\n\n')


    try:
        # Create path to save the velocity field
        save_path_particle_velocity = os.path.join(save_path, 'particle_velocity.npy')

        # Save the velocity field
        save_data(data = snapshot['vel'], file_path=save_path_particle_velocity)
        print("Particle velocity file ('particle_velocity.npy') saved successfully.")

    except:
        print('Error in saving the particle velocities. Skipping the step....\n\n')

#------------------------------ CREATE A DENSITY FIELD --------------------------------#

def get_density_field(snapshot: pynbody.simdict, mas: str, verbose: bool) -> np.ndarray:
    """
    Compute the density field or load an existing one.

    Parameters:
    - snapshot (pynbody.simdict): The loaded snapshot.
    - mas (str): The mass-assignment scheme. Options: 'NGP', 'CIC', 'TSC', 'PCS', 'gaussian'.
    - verbose (bool): Print information on progress.

    Returns:
    - np.ndarray: The computed or loaded density field.

    """
    # Load the density field if the user has it
    if not create_density:
        print('The density field is being loaded...')
        rho = load_data(own_density_path)
        print('Density field loaded successfully.')


    else:
        # Create path to save the density field
        save_path_density = os.path.join(save_path, 'density_field.npy')

        # Compute the density field
        print('Computing the density field...')
        rho = compute_density_field(snapshot=snapshot, grid_size=grid_size, box_size=box_size, mas=mas, verbose=verbose)
        print('Density field computed successfully.\n')
        
        # Save the density field
        save_data(data = rho, file_path=save_path_density)
        print("Density field ('density_field.npy') saved successfully.\n\n")
    
    return rho

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

#------------------------------ SMOOTH THE DENSITY FIELD --------------------------------#

def get_smoothed_field(input_field: np.ndarray, smoothing_scales: list[float], truncated_scales: list[str]) -> list[np.ndarray]:
    """
    Smooth the density field using the specified smoothing scales.

    Parameters:
    - input_field (np.ndarray): The input density field.
    - smoothing_scales (list[float]): List of smoothing scales.
    - truncated_scales (list[str]): List of truncated smoothing scale names.

    Returns:
    - list[np.ndarray]: List of smoothed density fields.

    """

    print('Smoothing the density field....')

    create_directory(os.path.join(f'{save_path}', 'smoothed_density_fields'))

    smoothened_rho = []

    for smth_scale in tqdm(smoothing_scales):
        smooth_rho = smooth_field(input_field=input_field, smoothing_scale=smth_scale, box_size = box_size, grid_size = grid_size)
        save_path_smooth = os.path.join(save_path, 'smoothed_density_fields', f'smoothed_density_field_{truncated_scales[smoothing_scales.index(smth_scale)]}.npy')
        save_data(data=smooth_rho, file_path=save_path_smooth)

        smoothened_rho.append(smooth_rho)
        
    print(f"Density field smoothed for all the smoothing scales and files saved successfully.")

    return smoothened_rho

#------------------------------ PLOT THE DENSITY FIELD --------------------------------#

def plot_density_field(input_field: np.ndarray, smoothing_scales: list[float], truncated_scales: list[str], 
                       projection: str, slice_thickness: list[int, int]) -> None:
    """
    Plot the smoothed density fields.

    Parameters:
    - input_field (np.ndarray): The smoothed density fields.
    - smoothing_scales (list[float]): List of smoothing scales.
    - truncated_scales (list[str]): List of truncated smoothing scale names.
    - projection (str): The projection type ('xy', 'yz', 'xz').
    - slice_thickness (list[int, int]): The range of slices to average over [Start index, End index].

    """
    # Get current directory
    current_dir = os.getcwd()
    # Make directory to store the plots
    create_directory(os.path.join(f'{current_dir}', 'density_plots'))

    for i, sm_scale in enumerate(smoothing_scales):
        save_sm_path = os.path.join(current_dir, 'density_plots')
        plot_field(input_field=input_field[i], 
                sm_scale=sm_scale,
                name_sm_scale=truncated_scales[i],
                projection=projection, # You can change the projection to 'yz' or 'xz'
                slice_indx=slice_thickness, # You can change the slice index: [Start index, End index]. This will average the density field over those slices.
                filepath=save_sm_path)

    print('Density field plots saved successfully.\n\n')

#------------------------------ CALCULATE TIDAL SHEAR TENSOR --------------------------------#

def calculate_tidal_tensor(smoothed_density_field: np.ndarray) -> None:
    """
    Calculate the tidal tensor and potential field.

    Parameters:
    - smoothed_density_field (np.ndarray): The smoothed density field.

    """

    print('Calculating the tidal tensor and potential field...') if calculate_potential else print('Calculating the tidal tensor...')

    # Make directory to store tidal tensor files
    create_directory(os.path.join(f'{save_path}', f'tidal_fields'))

    for i in tqdm(range(len(smoothed_density_field))):

        if calculate_potential:
            
            # Make directory to store potential field files
            create_directory(os.path.join(f'{save_path}', f'potential_field'))
            
            tidal_tensor, Grav_potential = calculate_tidal_tensor(density_field=smoothed_density_field[i], calculate_potential=True)

            save_path_tidal_tensor = os.path.join(save_path, 'tidal_fields', f'tidal_tensor_{truncated_scales[i]}.npy')
            save_path_tidal_potential = os.path.join(save_path, 'potential_field', f'potential_field_{truncated_scales[i]}.npy') if calculate_potential else None

            save_data(data=tidal_tensor, file_path=save_path_tidal_tensor)
            save_data(data=Grav_potential, file_path=save_path_tidal_potential) if calculate_potential else None

            if calculate_traceless:    
                
                traceless_tidal_shear = calculate_traceless_tidal_shear(tidal_tensor, grid_size)

                save_path_traceless = os.path.join(save_path, 'tidal_fields', f'traceless_tidal_shear_{truncated_scales[i]}.npy')

                save_data(data=traceless_tidal_shear, file_path=save_path_traceless)
        


        else:
            
            tidal_tensor = calculate_tidal_tensor(density_field=smoothed_density_field[i], calculate_potential=False)

            save_path_tidal_tensor = os.path.join(save_path, 'tidal_fields', f'tidal_tensor_{truncated_scales[i]}.npy')

            save_data(data=tidal_tensor, file_path=save_path_tidal_tensor)

            if calculate_traceless:
                
                traceless_tidal_shear = calculate_traceless_tidal_shear(tidal_tensor=tidal_tensor, grid_size=grid_size)

                save_path_traceless = os.path.join(save_path, 'tidal_fields', f'traceless_tidal_shear_{truncated_scales[i]}.npy')

                save_data(data=traceless_tidal_shear, file_path=save_path_traceless)




if __name__ == '__main__':

    snap, snap_header = get_snapshot(snapshot_path)

    extracted_values = extract_simulation_params(snapshot_header=snap_header)

    # Extract the box size
    box_size = int(extracted_values['boxsize'])

    save_simulation_properties(snapshot_header=snap_header, snapshot=snap)

    save_particle_positions_and_velocities(snapshot=snap, save_path=save_path)

    rho = get_density_field(snapshot=snap, mas='CIC', verbose=True)

    smth_scales, truncated_scales = extract_smoothing_scales(smoothing_scales)

    smoothened_rho = get_smoothed_field(input_field=rho, smoothing_scales=smth_scales, truncated_scales=truncated_scales)

    plot_density_field(input_field=smoothened_rho, smoothing_scales=smth_scales, truncated_scales=truncated_scales,
                       projection='xy', # Specify the projection of the box for classification overlay
                       slice_thickness=[0, grid_size//2] # Specify the range of slices to average over [Start index, End index]
                       )

    calculate_tidal_tensor(smoothed_density_field=smoothened_rho)

    print('All the calculations are done. Exiting the program...')

# Stop the timer
end_time = time.time()
print('Time taken to run the complete script:', end_time - start_time)

#----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#