
import numpy as np
import pynbody
from tqdm import tqdm
import MAS_library as MASL
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
from LSS_BB import *


# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, yn_smooth, \
        smoothing_scales, yn_potential, yn_traceless = read_input_file('input_params.txt')



print('\nInput parameters read succefully. Please wait...\n')

# Load the density field if the user has it
if not create_density:
    print('The density field is being loaded...')
    rho = load_data(own_density_path)
    print('Density field loaded successfully.')


else:
    print('Loading the snapshot...')
    snap = load_snapshot(snapshot_path)
    print('Snapshot loaded successfully.')

    save_path_density = os.path.join(save_path, 'density_field.npy')

    extracted_values = extract_simdict_values(snap.properties)
    box_size = extracted_values['boxsize']

    print('Computing the density field...')
    rho = compute_density_field(snap, grid_size, box_size, mas='CIC', verbose=True)
    print('Density field computed successfully.\n')
    
    save_data(rho, save_path_density)
    print("Density field ('density_field.npy') saved successfully.")


# Smoothing the density field
if yn_smooth == 'y':

    def extract_scales(input_scales):
        scales = input_scales.split()

        sm_scales = []

        for i in scales:
            sm_scales.append(float(i))
            sm_scales.sort()
        
        # Truncate the scales to 3 decimal places (used for naming the files later)
        truncated_scales = [str(int(float(x) * 1000))[:3] for x in sm_scales]
   
        return sm_scales, truncated_scales
    
    smth_scales = extract_scales(smoothing_scales)[0]
    truncated_scales = extract_scales(smoothing_scales)[1]
    

    print('Smoothing the density field....')

    smoothened_rho = []

    os.mkdir(f'{save_path}/smoothed_density_fields')

    for i, smth_scale in tqdm(enumerate(smth_scales)):
        smooth_rho = smooth_field(rho, smth_scale, box_size = box_size, grid_size = grid_size)
        save_path_smooth = os.path.join(save_path, 'smoothed_density_fields', f'smoothed_density_field_{truncated_scales[i]}.npy')
        save_data(smooth_rho, save_path_smooth)

        smoothened_rho.append(smooth_rho)
        
    print(f"Density field smoothed for all the smoothing scales and files saved successfully.") 

## PLOTS

# Make directory to store the plots

# Get current directory
current_dir = os.getcwd()
os.makedirs(f'{current_dir}/density_plots', exist_ok=True)

for i, sm_scale in enumerate(smth_scales):
    save_sm_path = os.path.join(current_dir, 'density_plots')
    plot_field(rho, truncated_scales[i], 'xy', sm_scale, int(grid_size/2), save_sm_path)

print('Density field plots saved successfully.')


# Calculate the tidal tensor and potential field
if not create_density:
    smoothened_rho = [rho]

# Make directory to store tidal tensor files
os.makedirs(os.path.join(f'{save_path}, tidal_fields'), exist_ok=True)


for i, smooth_rho in tqdm(enumerate(smoothened_rho)):
    if yn_potential:
        print('Calculating the tidal tensor and potential field...') if yn_potential else print('Calculating the tidal tensor...')
        tidal_tensor, Grav_potential = calculate_tidal_tensor(smooth_rho, calculate_potential=True)
        print('Tidal tensor and potential calculated successfully.') if yn_potential else print('Tidal tensor calculated successfully.')

        save_path_tidal_tensor = os.path.join(save_path, 'tidal_fields', f'tidal_tensor_{truncated_scales[i]}.npy')
        save_path_tidal_potential = os.path.join(save_path, 'tidal_fields', f'potential_field_{truncated_scales[i]}.npy') if yn_potential else None

        save_data(tidal_tensor, save_path_tidal_tensor)
        save_data(Grav_potential, save_path_tidal_potential) if yn_potential else None

        print("Tidal tensor and potential field saved successfully.")


        if yn_traceless:    
            print('Calculating the traceless tidal shear...')
            traceless_tidal_shear = calculate_traceless_tidal_shear(tidal_tensor, grid_size)
            print('Traceless tidal shear calculated successfully.')

            save_path_traceless = os.path.join(save_path, 'tidal_fields', f'traceless_tidal_shear_{truncated_scales[i]}.npy')

            save_data(traceless_tidal_shear, save_path_traceless)

            print("Traceless tidal shear saved successfully.")
    
        print('All the calculations are done. Exiting the program...') if i == (len(smoothened_rho) - 1) else None

    else:
        print('Calculating the tidal tensor...')
        tidal_tensor = calculate_tidal_tensor(smooth_rho, calculate_potential=False)
        print('Tidal tensor calculated successfully.')

        # Make directory to store tidal tensor files
        os.makedirs(f'{save_path}/tidal tensor', exist_ok=True)

        save_path_tidal_tensor = os.path.join(save_path, 'tidal_fields', f'tidal_tensor_{truncated_scales[i]}.npy')

        save_data(tidal_tensor, save_path_tidal_tensor)

        print("Tidal tensor saved successfully.")


        if yn_traceless:

            print('Calculating the traceless tidal shear...')
            traceless_tidal_shear = calculate_traceless_tidal_shear(tidal_tensor, grid_size)
            print('Traceless tidal shear calculated successfully.')

            save_path_traceless = os.path.join(save_path, 'tidal_fields', f'traceless_tidal_shear_{truncated_scales[i]}.npy')

            save_data(traceless_tidal_shear, save_path_traceless)

            print("Traceless tidal shear saved successfully.")

            print('All the calculations are done. Exiting the program...')

