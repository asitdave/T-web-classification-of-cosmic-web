# This script contains functions that help in computing the tidal shear tensor for a given density field 
# and classifies the structures based on the T-web classification scheme. 
# The library also provides functions to calculate the eigenvalues and eigenvectors of the tidal shear tensor, 
# as well as the traceless tidal shear tensor.

#-------------------------------------------------------------------------------------------------------------#

# Importing required libraries
import numpy as np
import pynbody
from tqdm import tqdm
import MAS_library as MASL
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import concurrent.futures
from numba import jit
import shutil


def read_input_file(file_path):
   with open("input_params.txt", "r") as file:
    for line in file:
      # Remove comments and leading/trailing spaces
      line = line.strip().split("#")[0]  # Split by comment symbol if needed

      # Check for keywords and extract values
      if line.startswith("Path of the snapshot file"):
        snapshot_path = line.split(":")[1].strip()
      elif line.startswith("Path where you want to save the results"):
        save_path = line.split(":")[1].strip()
      elif line.startswith("Enter the grid size"):
        grid_size = int(line.split(":")[1].strip())
      elif line.startswith("Create the density field"):
        create_density = line.split(":")[1].strip().lower() == "yes"
      elif line.startswith("Path to load the density field"):
        # Check if value exists (optional section)
        if ":" in line:
          density_path = None if (line.split(":")[1].strip()) == '' else int(line.split(":")[1].strip())
      elif line.startswith("Smooth density field"):
        smooth_density = line.split(":")[1].strip().lower() == "yes"
      elif line.startswith("Smoothing scales"):
        smoothing_scales = [float(x) for x in line.split(":")[1].strip().split()]
      elif line.startswith("Calculate potential field"):
        calculate_potential = line.split(":")[1].strip().lower() == "yes"
      elif line.startswith("Calculate traceless tidal tensor"):
        calculate_traceless_tidal = line.split(":")[1].strip().lower() == "yes"

    # Check for the possible incorrect/inconsistent input values
    # Additional logic based on retrieved values
    if density_path is None and create_density is False:
        raise ValueError("No density field provided!")
    
    if not os.path.exists(snapshot_path):
            raise ValueError('Please enter a valid snapshot path. File does not exist.')
    
    if not os.path.exists(save_path):
            raise ValueError('Please enter a valid file path to save the result. File does not exist.')
    
    try:
      grid_size = float(input('Enter the grid size: '))

      if grid_size <= 0:
        raise ValueError('Please enter a positive value.')
    
    except ValueError:
      raise ValueError('Please enter a valid grid size. (float)')
    
    if not os.path.exists(density_path):
            raise ValueError('Please enter a valid density field file path. File does not exist.')

    
    return [snapshot_path, save_path, grid_size, create_density, density_path, smooth_density, smoothing_scales, calculate_potential, calculate_traceless_tidal]



def load_snapshot(snapshot_path):
    """
    Load the cosmological simulation snapshot.

    Parameters:
      - snapshot_path: str
        The path to the snapshot file.

    Returns:
      - snap: pynbody.snapshot object
        The loaded cosmological simulation snapshot.
    """

    snap = pynbody.load(snapshot_path)

    header = snap.properties
    print("Snapshot properties: ", header, '\n')

    print("Loadable keys:", snap.loadable_keys(), '\n')

    return snap


def extract_simdict_values(simdict):
  """Extracts values from a pynbody SimDict in float format.

  Args:
      simdict: A pynbody.simdict.SimDict object.

  Returns:
      A dictionary containing the following keys and values in float format:
          boxsize: (float) Box size in Mpc/h
          time: (float) Time in seconds
          a: (float) Scale factor
          h: (float) Hubble parameter
          omega_m: (float) Matter density parameter
          omega_l: (float) Dark energy density parameter
  """

  results = {}
  for key, value in simdict.items():
    if key in ['omegaM0', 'omegaL0', 'a', 'h']:
      results[key] = float(value)
    else:
      results[key] = float(str(value).split()[0])

  return results


def load_data(file_path):
    """
    Load the data from the specified file path.

    Parameters:
    - file_path: str
      The file path where the data is stored.

    Returns:
    - data: numpy.ndarray
      The loaded data.
    """
    
    if type(file_path) != str:
      return ValueError("The file path should be a string.")

    try:
      data = np.load(file_path)

    except ValueError:
      print("Error loading the data from the file path.")

    return data


def create_directory(directory_path):
  """
  Creates a directory, overwriting any existing directory with the same name.

  Args:
      directory_path: str
          Path to the directory to create.

  Returns:
      None
  """
  try:
    os.makedirs(directory_path)  # Attempt to create directories recursively

  except OSError as e:
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
      # Overwrite existing directory if it's a directory
      print(f"Warning: Overwriting existing directory: {directory_path}")
      # You can add additional logic here to prompt for confirmation before overwriting
      shutil.rmtree(directory_path)  # Remove the existing directory
      os.makedirs(directory_path)  # Create the directory again
    else:
      raise e  # Re-raise the original error for other OSError cases



def save_data(data, file_path):
    """
    Save the given data to the specified file path.

    Parameters:
    - data: numpy.ndarray
      The data to be saved.
    - file_path: str
      The file path where the data will be saved.
    """
    
    if type(file_path) != str:
      return ValueError("The file path should be a string.")

    try:
      np.save(file_path, data)

    except ValueError:
      print("Error saving the data to the file path.")


def compute_density_field(snapshot, grid_size, box_size, mas='CIC', verbose=True):
    """
    Compute the density field from a cosmological simulation snapshot.

    Parameters:
      - snapshot: pynbody.snapshot object
        The cosmological simulation snapshot.
      - grid_size: int
        The size of the 3D grid for the density field.
      - box_size: float
        The size of the simulation box in Mpc/h.
      - mas: str, optional (default='CIC')
        The mass-assignment scheme. Options: 'NGP', 'CIC', 'TSC', 'PCS', 'gaussian'.
      - verbose: bool, optional (default=True)
        Print information on progress.

    Returns:
    - delta: numpy.ndarray
      The computed density field representing density contrast in each voxel.
    """
    pos = snapshot["pos"] # Position of particles

    # Construct a 3D grid for the density field
    delta = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    MASL.MA(pos, delta, box_size, mas, verbose=verbose) # Generate the density field
    
    # Calculate the density contrast
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0 

    return delta



def smooth_field(field, smoothing_scale, box_size, grid_size):
    """
    Smooth the given field using Gaussian filtering.

    Parameters:
    - field: numpy.ndarray
      The field to be smoothed.
    - smoothing_scale: float
      The scale of Gaussian smoothing.
    - box_size: float
      The size of the simulation box in Mpc/h.
    - grid_size: int
      The size of the 3D grid for the density field.

    Returns:
    - smoothed_field: numpy.ndarray
      The smoothed field.
    """
    sigma = smoothing_scale
    pixel_scale = box_size / grid_size
    sigma_pixels = sigma / pixel_scale

    # mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    return gaussian_filter(field, sigma=sigma_pixels, mode='wrap', cval=0.0)



def plot_field(field, title, dim, sm_scale, slice=0):
    """
    Plot the given field.

    Parameters:
    - field: numpy.ndarray
      The field to be plotted.
    - title: str
      The title of the plot.
    - slice: int, optional (default=0)
      The slice of the field to be plotted.
    - axis: int, optional (default=2)
      The axis along which the slice is taken.
    """
    eps=1e-15 # so that log doesn't get a value 0
    N = None

    # Plot for zero smoothening density field
    plt.figure(figsize = (2,2), dpi = 200)
    delplot1=np.log10(field+1+eps)

    if dim == ('yz' or 'zy'):
      slic1=np.mean(delplot1[N:slice+N, :, :],axis=0)

    if dim == ('xz' or 'zx'):
      slic1=np.mean(delplot1[:, N:slice+N, :],axis=1)
    
    if dim == ('xy' or 'yx'):
      slic1=np.mean(delplot1[:, :, N:slice+N],axis=2)

    plt.imshow(slic1, cmap = 'inferno')
    plt.axis('off')
    plt.title(r'$R_s = {sm_scale}~h^{-1} Mpc$'.format(sm_scale=str(sm_scale)))
    plt.show()



def calculate_tidal_tensor(density_field, calculate_potential=False):
    """
    Calculate the tidal tensor from the given density field.

    Parameters:
    - density_field: numpy.ndarray
      The density field.
    - calculate_potential: bool, optional (default=False)
      Calculate the potential field.

    Returns:
    - tidal_tensor: numpy.ndarray
      Tidal tensor (3x3 matrix) for each voxel in the simulation box.
    """

    # Fast Fourier Transform of the density field
    density_field_fft = np.fft.fftn(density_field)
    shape = density_field.shape[0]

    # Generate the k-space grid
    kx_modes = np.fft.fftfreq(shape)
    ky_modes = np.fft.fftfreq(shape)
    kz_modes = np.fft.fftfreq(shape)
    kx, ky, kz = np.meshgrid(kx_modes, ky_modes, kz_modes, indexing="ij")

    # Calculate for all permutations
    kx2 = np.multiply(kx, kx)
    ky2 = np.multiply(ky, ky)
    kz2 = np.multiply(kz, kz)
    kxy = np.multiply(kx, ky)
    kxz = np.multiply(kx, kz)
    kyz = np.multiply(ky, kz)
    k_sq = kx2 + ky2 + kz2 # Calculate the square of k

    # Calculate the tidal tensor in Fourier space
    potential_k = -np.divide(density_field_fft, k_sq, where=k_sq != 0)

    # Calculate the potential
    if calculate_potential:
      potential = np.fft.ifft(potential_k)

    # Calculate the tidal tensor
    tidal_tensor = np.zeros((3, 3) + density_field.shape, dtype=np.complex128)
    tk00 = -potential_k * kx2
    tk01 = -potential_k * kxy
    tk02 = -potential_k * kxz
    tk11 = -potential_k * ky2
    tk12 = -potential_k * kyz
    tk22 = -potential_k * kz2

    # Inverse Fourier Transform to get the tidal tensor
    tt00 = np.fft.ifftn(tk00)
    tt01 = np.fft.ifftn(tk01)
    tt02 = np.fft.ifftn(tk02)
    tt12 = np.fft.ifftn(tk12)
    tt11 = np.fft.ifftn(tk11)
    tt22 = np.fft.ifftn(tk22)

    # Assembling the tensor field (assigning the values to symmetric counterparts)
    tidal_tensor[0, 0, ...] = tt00
    tidal_tensor[1, 0, ...] = tt01
    tidal_tensor[0, 1, ...] = tt01
    tidal_tensor[1, 1, ...] = tt11
    tidal_tensor[0, 2, ...] = tt02
    tidal_tensor[2, 0, ...] = tt02
    tidal_tensor[2, 2, ...] = tt22
    tidal_tensor[1, 2, ...] = tt12
    tidal_tensor[2, 1, ...] = tt12
    tidal_tensor = tidal_tensor.real

    if calculate_potential:
      return tidal_tensor, potential
    
    else:
      return tidal_tensor


def make_traceless(matrix):
    """
    Make the given matrix traceless.

    Parameters:
    - matrix: numpy.ndarray
      The matrix to be made traceless.

    Returns:
    - traceless_matrix: numpy.ndarray
      The traceless matrix.
    """
    traceless_matrix = matrix - (np.trace(matrix) / 3) * np.identity(3)
    return traceless_matrix



def calculate_traceless_tidal_shear(tidal_tensor, grid_size):
    """
    Calculate the traceless tidal shear from the given tidal tensor.

    Parameters:
    - tidal_tensor: numpy.ndarray
      The tidal tensor.
    - grid_size: int
      The size of the 3D grid for the density field.

    Returns:
    - tidal_shear: numpy.ndarray
      The traceless tidal shear.
    """
    tid = tidal_tensor.reshape(grid_size, grid_size, grid_size, 3, 3)
    traceless = []
    for i in tqdm(range(len(tid))):
        traceless.append(make_traceless(tid[i]))
    return np.array(traceless).reshape(grid_size, grid_size, grid_size, 3, 3)


def save_data(data, file_path):
    """
    Save the given data to the specified file path.

    Parameters:
    - data: numpy.ndarray
      The data to be saved.
    - file_path: str
      The file path where the data will be saved.
    """
    
    if type(file_path) != str:
      return ValueError("The file path should be a string.")

    try:
      np.save(file_path, data)

    except ValueError:
      print("Error saving the data to the file path.")








def load_all_npy_files(folder_path):
  """
  Loads all files with the .npy extension from a folder using a loop.

  Args:
      folder_path: str
          Path to the folder containing the NumPy files.

  Returns:
      list:
          A list of loaded NumPy arrays from the folder.
  """

  data_list = []
  for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
      filepath = os.path.join(folder_path, filename)
      data = np.load(filepath)
      data_list.append(data)

    else:
       print('No .npy files found in the folder.')

  return data_list



# Define the JIT-compiled function
@jit(nopython=True)
def calculate_eigenvalues_and_vectors(tidal_shear_tensor, grid_size):
    eigenvalues = np.zeros((grid_size, grid_size, grid_size, 3))
    eigenvectors = np.zeros((grid_size, grid_size, grid_size, 3, 3))

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                eigenvalues[i, j, k], eigenvectors[i, j, k] = np.linalg.eig(tidal_shear_tensor[i, j, k])
                
                # Sort the eigenvalues and corresponding eigenvectors in descending order
                sorted_indices = np.argsort(eigenvalues[i, j, k])[::-1]
                eigenvalues[i, j, k] = eigenvalues[i, j, k][sorted_indices]
                eigenvectors[i, j, k] = eigenvectors[i, j, k][sorted_indices]

    return eigenvalues, eigenvectors



def classify_structure(eigenvalues):
    ''' 
    Uses T-web classification scheme.
    Classifies each voxel as a part of Void, Sheet, Filament, or Node
    based on the eigenvalues of the tidal tensor.
    '''

    num_positive = np.sum(eigenvalues > 0)
    num_negative = np.sum(eigenvalues < 0)

    if num_positive == 0:
        return 0 # "Void"
    elif num_positive == 1 and num_negative == 2:
        return 1 # "Sheet"
    elif num_positive == 2 and num_negative == 1:
        return 2 # "Filament"
    elif num_positive == 3:
        return 3 # "Cluster"


def calculate_volume_fraction(classification_matrix, label):
    """
    Calculate the volume fraction of a --specific label-- in a 3D classification matrix.

    Parameters:
    - classification_matrix (numpy.ndarray): 3D array containing voxel labels.
    - label (int): The label for which the volume fraction is calculated.

    Returns:
    float: The volume fraction of the specified label in the matrix.
    """
    return np.sum(classification_matrix.flatten() == label) / np.prod(classification_matrix.shape)


def calculate_volume_fractions(classification_matrix):
    """
    Calculate the volume fractions for --multiple labels-- in a 3D classification matrix.

    Parameters:
    - classification_matrix (numpy.ndarray): 3D array containing voxel labels.
    - num_labels (numpy.ndarray or list): Array or list of labels for which volume fractions are calculated.

    Returns:
    numpy.ndarray: Array of volume fractions corresponding to each label.
    """

    num_labels = np.arange(4) # Labels: 0 = Void, 1 = Sheet, 2 = Filament, 3 = Cluster

    volume_fractions = np.zeros_like(num_labels, dtype=float)

    for i, label in enumerate(num_labels):
        volume_fractions[i] = calculate_volume_fraction(classification_matrix, label)
    
    return volume_fractions



def slice_density_field(density_field, slice_index):
    """
    Extracts a 2D slice from a 3D density field and applies logarithmic transformation.

    Parameters:
    - density_field (numpy.ndarray): 3D array representing the density field.
    - slice_index (int): Index of the slice to be extracted along the z-axis.

    Returns:
    - numpy.ndarray: 2D array representing the logarithmic transformation of the specified slice.

    Notes:
    - The logarithmic transformation is applied using the formula np.log10(density_field + 1 + eps),
      where eps is a small constant (1e-15) to prevent taking the logarithm of zero.

    Example:
    >>> density_field = np.random.rand(5, 5, 5)
    >>> slice_index = 2
    >>> result = slice_density_field(density_field, slice_index)
    """
    eps = 1e-15  # so that log doesn't get a value 0
    N = 0

    delplot1 = np.log10(density_field + 1 + eps)
    d_field = delplot1[:, :, slice_index]
    
    return d_field




def get_pos(structure, projection, slice_index, classification, grid_size=512, box_size=100):
    
    """
    Extracts positions of points in a 2D slice based on the specified structure, projection, and classification.

    Parameters:
    - structure (str): A string representing the structure type ('v', 's', 'f', 'n').
    - projection (str): A string representing the projection type ('xy', 'yx', 'yz', 'zy', 'zx', 'xz').
    - slice_index (int): Index of the slice to be extracted.
    - classification (numpy.ndarray): 3D array representing the classification of points.
    - grid_size (int): Size of the grid in the 3D array (default is 512).
    - box_size (int): Size of the box in physical units (default is 100).

    Returns:
    - numpy.ndarray: 2D array representing the positions of points in the specified slice.

    Notes:
    - Points are classified into different structures based on the structure parameter.
    - The projection parameter determines the orientation of the slice (e.g., 'xy', 'yz', 'zx').
    - The classification parameter is a 3D array where points are classified into different structures.
    - The positions are returned in physical units, scaled based on the grid size and box size.
    - The function prints 'ValueError' if invalid structure or projection is provided.

    Example:
    >>> structure = 'v'
    >>> projection = 'xy'
    >>> slice_index = 2
    >>> classification = np.random.randint(0, 4, size=(512, 512, 512))
    >>> result = get_pos(structure, projection, slice_index, classification)
    """
    
    if str(structure)[0].lower() == 'v':

        if (str(projection) == 'xy' or 'yx'):
            mask_r, mask_c = np.where(classification[:, :, slice_index] == 0)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_c, mask_r))
            return position
        
        elif (str(projection) == 'yz' or 'zy'):
            mask_r, mask_c = np.where(classification[slice_index, :, :] == 0)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        elif (str(projection) == 'zx' or 'xz'):
            mask_r, mask_c = np.where(classification[:, slice_index, :] == 0)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        else: 
            print('ValueError')
            
            
    elif str(structure)[0].lower() == 's':
        
        if (str(projection) == 'xy' or 'yx'):
            mask_r, mask_c = np.where(classification[:, :, slice_index] == 1)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_c, mask_r))
            return position
        
        elif (str(projection) == 'yz' or 'zy'):
            mask_r, mask_c = np.where(classification[slice_index, :, :] == 1)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        elif (str(projection) == 'zx' or 'xz'):
            mask_r, mask_c = np.where(classification[:, slice_index, :] == 1)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        else: 
            print('ValueError')
            
            
    elif str(structure)[0].lower() == 'f':
        
        if (str(projection) == 'xy' or 'yx'):
            mask_r, mask_c = np.where(classification[:, :, slice_index] == 2)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_c, mask_r))
            return position
        
        elif (str(projection) == 'yz' or 'zy'):
            mask_r, mask_c = np.where(classification[slice_index, :, :] == 2)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        elif (str(projection) == 'zx' or 'xz'):
            mask_r, mask_c = np.where(classification[:, slice_index, :] == 2)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        else: 
            print('ValueError')
            
            
    elif str(structure)[0].lower() == 'n':
        
        if (str(projection) == 'xy' or 'yx'):
            mask_r, mask_c = np.where(classification[:, :, slice_index] == 3)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_c, mask_r))
            return position
        
        elif (str(projection) == 'yz' or 'zy'):
            mask_r, mask_c = np.where(classification[slice_index, :, :] == 3)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        elif (str(projection) == 'zx' or 'xz'):
            mask_r, mask_c = np.where(classification[:, slice_index, :] == 3)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position
        
        else: 
            print('ValueError')
        
    else:
        print('ValueError')
        


def get_class_env_pos(classification, slice_index):
    
    """
    Extracts positions of points for all environment types ('v', 's', 'f', 'n') in a 2D slice.

    Parameters:
    - classification (numpy.ndarray): 3D array representing the classification of points.
    - slice_index (int): Index of the slice to be extracted.

    Returns:
    - list: List containing 2D arrays representing positions of points for each environment type.

    Notes:
    - Uses the get_pos function to extract positions for each environment type.
    - The classification parameter is a 3D array where points are classified into different structures.
    - The slice_index parameter determines the index of the slice to be extracted.
    - Positions are returned in physical units, scaled based on the default grid size (512) and box size (100).

    Example:
    >>> classification = np.random.randint(0, 4, size=(512, 512, 512))
    >>> slice_index = 2
    >>> result = get_class_env_pos(classification, slice_index)
    """
    
    envs = ['v', 's', 'f', 'n']
    
    all_env_pos = []
    
    for env in envs:
        env_pos = get_pos(env, 'xy', slice_index, classification)
        all_env_pos.append(env_pos)
    
    return all_env_pos