

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import os
from LSS_BB import *


# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, yn_smooth, \
        smoothing_scales, yn_potential, yn_traceless = read_input_file('input_params.txt')

with open('input_params.txt', 'r') as f:
    lines = f.readlines()

    for line in lines:
        if line.startswith('box_size'):
            box_size = float(line.split(':')[1].strip())


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

classification_path = os.path.join(save_path, 'Classification matrices')

classification_files = sorted([name for name in os.listdir(classification_path) if name.startswith('classification_matrix')])
total_files = len(classification_files)


print('Number of files found:', total_files)

print('Loading the classification matrix files....')

# Load the classification matrix files
classifications = load_all_npy_files(classification_path)

print('Classification matrix files loaded successfully.\n')

new_directory_name = 'Classification analysis plots'
new_directory_path = os.path.join(classification_path, new_directory_name)
create_directory(new_directory_path)


labels = ['Void', 'Sheets', 'Filaments', 'Clusters']

vol_fracs = []

for class_i in tqdm(range(len(classifications))):
    vol_frac = calculate_volume_fractions(class_i)
    vol_fracs.append(vol_frac)

vol_fracs = np.array(vol_fracs)


# Plot the volume fractions
plt.figure(figsize = (5,5), dpi = 300)
for i in range(len(vol_fracs.T)):
    plt.semilogx(smth_scales, vol_fracs[:, i], label = labels[i])

plt.axhline(0.43, ls = '--', lw = 1, color = 'black', alpha = 0.4) # Attains gaussian random field
plt.axhline(0.072, ls = '--', lw = 1, color = 'black', alpha = 0.4) # Attains gaussian random field
# plt.axvline(0.54, ls = '--', lw = 1, color = 'black', alpha = 0.8)

plt.xlabel('$R_s~[h^{-1}~ Mpc] $')
plt.ylabel('Volume fraction')
plt.tick_params(axis='both',  which='both',  left=True, right=True, top = True, bottom = True, direction = 'in', labelsize = 7)
plt.legend(bbox_to_anchor = (1, 0.6), fontsize = 9, fancybox = True, handlelength = 1)
plt.savefig(os.path.join(classification_path, 'Volume_fractions_vs_Rs.png'))









