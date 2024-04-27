import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import itertools
import os
import shutil
from LSS_BB import *

while True:
    try:
        classification_path = str(input('Give the path where all classification matrix files (.npy) are stored: '))
        if not os.path.exists(classification_path):
            raise ValueError('Please enter a valid file path.')
        break
    except ValueError as e:
        print(e)


# Read the input file
snapshot_path, save_path, grid_size, \
    create_density, own_density_path, yn_smooth, \
        smoothing_scales, yn_potential, yn_traceless = read_input_file('input_params.txt')

def extract_scales(input_scales):
    scales = input_scales.split()

    sm_scales = []

    for i in scales:
        sm_scales.append(float(i))
        sm_scales.sort()
    
    truncated_scales = [str(int(float(x) * 1000))[:3] for x in sm_scales]

    return sm_scales, truncated_scales

smth_scales = extract_scales(smoothing_scales)[0]
truncated_scales = extract_scales(smoothing_scales)[1]


classification_files = [name for name in os.listdir(classification_path) if name.startswith('classification_matrix')]
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

plt.figure(figsize = (5,5), dpi = 300)
for i in range(len(vol_fracs.T)):
    plt.semilogx(smth_scales, vol_fracs[:, i], label = labels[i])

plt.axhline(0.43, ls = '--', lw = 1, color = 'black', alpha = 0.4)
plt.axhline(0.072, ls = '--', lw = 1, color = 'black', alpha = 0.4)
plt.axvline(0.54, ls = '--', lw = 1, color = 'black', alpha = 0.8)

plt.xlabel('$R_s~[h^{-1}~ Mpc] $')
plt.ylabel('Volume fraction')
plt.tick_params(axis='both',  which='both',  left=True, right=True, top = True, bottom = True, direction = 'in', labelsize = 7)
# plt.xticks(fontsize = 10)
plt.legend(bbox_to_anchor = (1, 0.6), fontsize = 9, fancybox = True, handlelength = 1)
plt.savefig(os.path.join(classification_path, 'Volume_fractions_vs_Rs.png'))







