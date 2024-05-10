#------------------------------------------------------------------------------------------------------------------------------------------
# - This script computes the density field, tidal tensor, potential field and traceless tidal shear tensor for a given pynbody.snapshot. 
# - The user can also smooth the density field using a Gaussian filter.
# - The user can also choose to compute the potential field and traceless tidal.
#------------------------------------------------------------------------------------------------------------------------------------------

# Importing the necessary libraries
from tqdm import tqdm
import os
from LSS_BB import *
import time

# Start the timer
start_time = time.time()

#------------------------------ READ INPUT FILE ----------------------------------#

# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, yn_smooth, \
        smoothing_scales, yn_potential, yn_traceless = read_input_file('input_params.txt')

print('\nInput parameters read succefully. Please wait...\n')


#------------------------------ LOAD THE SNAPSHOT --------------------------------#

print('Loading the snapshot...')
snap, header = load_snapshot(snapshot_path)
print('Snapshot loaded successfully.')

# Extract the simulation properties
extracted_values = extract_simdict_values(simdict=snap.properties)
box_size = extracted_values['boxsize']

# Save the simulation properties to a file
with open('simulation_properties.txt', "w") as f:
    f.write('-------------------------------------------------------------\n')
    f.write(' The file contains the simulation properties and parameters \n')
    f.write('-------------------------------------------------------------\n\n')
    
    f.write('Box size: ' + str(header['boxsize']) + '\n')
    f.write('Omega_M0: ' + str(header['omegaM0']) + '\n')
    f.write('Omega_L0: ' + str(header['omegaL0']) + '\n')
    f.write('a: ' + str(header['a']) + '\n')
    f.write('h: ' + str(header['h']) + '\n')
    f.write('time: ' + str(header['time']) + '\n\n')
    
    mean_mass = snap['mass'].mean()
    f.write('Mass of each particle: ' + str(mean_mass) + ' * ' + str(mean_mass.units) + '\n')
    
    total_particles = snap['pos'].shape[0]
    f.write('Total number of particles: ' + str(total_particles) + '\n')

print('Simulation properties saved successfully. Please check the "simulation_properties.txt" file in cwd.\n\n')


#------------------------------ CREATE A DENSITY FIELD --------------------------------#

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
    rho = compute_density_field(snapshot=snap, grid_size=grid_size, box_size=box_size, mas='CIC', verbose=True)
    print('Density field computed successfully.\n')
    
    # Save the density field
    save_data(data = rho, file_path=save_path_density)
    print("Density field ('density_field.npy') saved successfully.")

#------------------------------ SMOOTH THE DENSITY FIELD --------------------------------#
if yn_smooth:
    
    # Extract the smoothing scales
    smth_scales = extract_scales(smoothing_scales)[0] # Extract the smoothing scales (float)
    truncated_scales = extract_scales(smoothing_scales)[1] # Truncate the scales to remove the decimal point
    
    print('Smoothing the density field....')

    create_directory(os.path.join(f'{save_path}', 'smoothed_density_fields'))

    smoothened_rho = []

    for smth_scale in tqdm(smth_scales):
        smooth_rho = smooth_field(input_field=rho, smoothing_scale=smth_scale, box_size = box_size, grid_size = grid_size)
        save_path_smooth = os.path.join(save_path, 'smoothed_density_fields', f'smoothed_density_field_{truncated_scales[smth_scales.index(smth_scale)]}.npy')
        save_data(smooth_rho, save_path_smooth)

        smoothened_rho.append(smooth_rho)
        
    print(f"Density field smoothed for all the smoothing scales and files saved successfully.")

else:
    print('You chose not to smooth the density field!\n\n')
    
    yn_continue = input('Do you want to continue with the calculations? (y/n): ')

    if yn_continue == 'n':
        print('Exiting the program....')
        exit()
    
    else:
        pass


#------------------------------ PLOT THE DENSITY FIELD --------------------------------#

# Get current directory
current_dir = os.getcwd()
# Make directory to store the plots
create_directory(os.path.join(f'{current_dir}', 'density_plots'))

for i, sm_scale in enumerate(smth_scales):
    save_sm_path = os.path.join(current_dir, 'density_plots')
    plot_field(input_field=rho, sm_scale=sm_scale,
                dim='xy', slice=int(grid_size/2), slice_thickness=5, 
                name_sm_scale=truncated_scales[i], filepath=save_sm_path)

print('Density field plots saved successfully.')


#------------------------------ CALCULATE TIDAL SHEAR TENSOR --------------------------------#

# Make directory to store tidal tensor files
create_directory(os.path.join(f'{save_path}, tidal_fields'))

for i, smooth_rho in tqdm(enumerate(smoothened_rho)):

    if yn_potential:

        tidal_tensor, Grav_potential = calculate_tidal_tensor(smooth_rho, calculate_potential=True)

        save_path_tidal_tensor = os.path.join(save_path, 'tidal_fields', f'tidal_tensor_{truncated_scales[i]}.npy')
        save_path_tidal_potential = os.path.join(save_path, 'tidal_fields', f'potential_field_{truncated_scales[i]}.npy') if yn_potential else None

        save_data(tidal_tensor, save_path_tidal_tensor)
        save_data(Grav_potential, save_path_tidal_potential) if yn_potential else None

        if yn_traceless:    
            
            traceless_tidal_shear = calculate_traceless_tidal_shear(tidal_tensor, grid_size)

            save_path_traceless = os.path.join(save_path, 'tidal_fields', f'traceless_tidal_shear_{truncated_scales[i]}.npy')

            save_data(traceless_tidal_shear, save_path_traceless)
    
        print('All the calculations are done. Exiting the program...') if i == (len(smoothened_rho) - 1) else None


    else:
        
        tidal_tensor = calculate_tidal_tensor(smooth_rho, calculate_potential=False)

        save_path_tidal_tensor = os.path.join(save_path, 'tidal_fields', f'tidal_tensor_{truncated_scales[i]}.npy')

        save_data(tidal_tensor, save_path_tidal_tensor)

        if yn_traceless:
            
            traceless_tidal_shear = calculate_traceless_tidal_shear(tidal_tensor, grid_size)

            save_path_traceless = os.path.join(save_path, 'tidal_fields', f'traceless_tidal_shear_{truncated_scales[i]}.npy')

            save_data(traceless_tidal_shear, save_path_traceless)

            print('All the calculations are done. Exiting the program...')


# Stop the timer
end_time = time.time()
print('Time taken to run the complete script:', end_time - start_time)

#----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#