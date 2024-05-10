#-------------------------------------------------------------------------------------------------------------#
# This script contains functions that help in computing the tidal shear tensor for a given density field 
# and classifies the structures based on the T-web classification scheme. 
# It provides functions to calculate the eigenvalues and eigenvectors of the tidal shear tensor, 
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
from numba import jit
import shutil

#-------------------------------------------------------------------------------------------------------------#

def read_input_file(file_path: str) -> list:
   
   # Initialize variables to store extracted values
    snapshot_path = ""
    save_path = ""
    create_density = True
    density_path = None
    smooth_density = False
    smoothing_scales = None
    grid_size = 0
    calculate_potential = False
    calculate_traceless_tidal = False


    with open(file_path, "r") as file:
      for line in file:
          # Remove comments and leading/trailing spaces
          line = line.strip().split("#")[0]  # Split by comment symbol if needed

          # Check for keywords and extract values
          if line.startswith("Path to the snapshot file"):
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

          if grid_size <= 0:
            raise ValueError('Please enter a positive value.')
              
          for path in [snapshot_path, save_path]:
              if not os.path.exists(path):
                  raise ValueError(f"Please enter a valid path. {path} does not exist.")

          if density_path is not None:
              if not os.path.exists(density_path):
                  raise ValueError(f"Please enter a valid path. {density_path} does not exist.")

      return [snapshot_path, save_path, grid_size, create_density, density_path, smooth_density, smoothing_scales, calculate_potential, calculate_traceless_tidal]

#-------------------------------------------------------------------------------------------------------------#

def load_snapshot(snapshot_path: str) -> pynbody.snapshot:
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

    return snap, header

#-------------------------------------------------------------------------------------------------------------#

def extract_simdict_values(simdict: pynbody.simdict.SimDict) -> dict:
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

#-------------------------------------------------------------------------------------------------------------#

def extract_scales(input_scales: list) -> tuple:
    """
    Extracts and processes scales from a list of input smoothing scales.

    Parameters:
    - input_scales: list
      A list containing scales to be extracted and processed.

    Returns:
    - sm_scales: list
      A list of smoothing scales converted to float.
    - truncated_scales: list
      A list of smoothing scales with decimal points removed.
    """
    # Sort the input scales in ascending order
    scales = sorted(input_scales)

    # Initialize lists to store processed scales
    truncated_scales = []
    sm_scales = []
    
    # Process each scale
    for scale in scales:
        # Convert scale to float and append to sm_scales
        sm_scales.append(float(scale))

        # Remove decimal points from scale and append to truncated_scales
        str_split = str(scale).split('.')
        truncated_scales.append(''.join(str_num for str_num in str_split))

    return sm_scales, truncated_scales

#-------------------------------------------------------------------------------------------------------------#

def load_data(file_path: str) -> np.ndarray:
    """
    Load the data from the specified file path.

    Parameters:
    - file_path: str
      The file path where the data is stored.

    Returns:
    - data: numpy.ndarray
      The loaded data.
    """

    try:
      loaded_data = np.load(file_path)

    except ValueError:
      print("Error loading the data from the file path.")

    return loaded_data

#-------------------------------------------------------------------------------------------------------------#

def create_directory(directory_path: str) -> None:
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

#-------------------------------------------------------------------------------------------------------------#

def save_data(data: np.ndarray, file_path: str) -> None:
    """
    Save the given data to the specified file path.

    Parameters:
    - data: numpy.ndarray
      The data to be saved.
    - file_path: str
      The file path where the data will be saved.
    """

    try:
      np.save(file_path, data)

    except ValueError:
      print("Error saving the data to the file path.")

#-------------------------------------------------------------------------------------------------------------#

def compute_density_field(snapshot: pynbody.snapshot, grid_size: int, box_size, mas='CIC', verbose=True):
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

#-------------------------------------------------------------------------------------------------------------#

def smooth_field(input_field: np.ndarray, smoothing_scale: float, box_size, grid_size: int):
    """
    Smooth the given field using Gaussian filtering.

    Parameters:
    - input_field: numpy.ndarray
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
    return gaussian_filter(input_field, sigma=sigma_pixels, mode='wrap', cval=0.0)

#-------------------------------------------------------------------------------------------------------------#

def plot_field(input_field: np.ndarray, sm_scale: float, dim: str, slice: int, slice_thickness: int, name_sm_scale: str, filepath: str):
    """
    Plot the given field.

    Parameters:
    - input_field: numpy.ndarray
      The field to be plotted.
    - name_sm_scale: str
      Smoothing scale for the plot (used for saving the plot).
    - dim: str
      The dimension of the field ('xy', 'yz', 'zx').
    - sm_scale: float
      The smoothing scale.
    - slice: int, optional (default= half grid size)
      The slice of the field to be plotted.
    - filepath: str
      The file path where the plot will be saved.
    - slice_thickness: int
      The thickness of the slice.
    """
    eps=1e-15 # so that log doesn't get a value 0
    N = slice_thickness

    # Plot for zero smoothening density field
    plt.figure(figsize = (2,2), dpi = 200)
    delplot1=np.log10(input_field+1+eps)

    if dim == ('yz' or 'zy'):
      slic1=np.mean(delplot1[N:slice+N, :, :],axis=0)

    if dim == ('xz' or 'zx'):
      slic1=np.mean(delplot1[:, slice-N:slice+N, :],axis=1)
    
    if dim == ('xy' or 'yx'):
      slic1=np.mean(delplot1[:, :, slice-N:slice+N],axis=2)

    plt.imshow(slic1, cmap = 'inferno')
    plt.axis('off')
    plt.title(r'$R_s = $' + f'{sm_scale}' + r'$~h^{-1} Mpc$')
    
    save_path = os.path.join(filepath, f'{dim}_plane_{name_sm_scale}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

#-------------------------------------------------------------------------------------------------------------#

def calculate_tidal_tensor(density_field: np.ndarray, calculate_potential=False) -> np.ndarray:
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
    k_modes = np.fft.fftfreq(shape)
    kx, ky, kz = np.meshgrid(k_modes, k_modes, k_modes, indexing="ij")

    # Calculate the square of k
    k_sq = kx**2 + ky**2 + kz**2

    # Calculate the tidal tensor in Fourier space
    with np.errstate(divide='ignore', invalid='ignore'):
        potential_k = -np.divide(density_field_fft, k_sq, where=k_sq != 0)

    # Calculate the potential if required
    if calculate_potential:
        potential = np.fft.ifftn(potential_k).real

    # Calculate the components of the tidal tensor in Fourier space
    tk_components = -potential_k * np.array([kx**2, kx*ky, kx*kz, ky**2, ky*kz, kz**2])

    # Inverse Fourier Transform to get the tidal tensor components
    tidal_tensor_components = np.fft.ifftn(tk_components)

    # Assemble the tensor field (assigning the values to symmetric counterparts)
    tidal_tensor = np.zeros((3, 3) + density_field.shape, dtype=np.float64)
    tidal_tensor[0, 0, ...] = tidal_tensor_components[0].real
    tidal_tensor[1, 0, ...] = tidal_tensor_components[1].real
    tidal_tensor[0, 1, ...] = tidal_tensor_components[1].real
    tidal_tensor[1, 1, ...] = tidal_tensor_components[3].real
    tidal_tensor[0, 2, ...] = tidal_tensor_components[2].real
    tidal_tensor[2, 0, ...] = tidal_tensor_components[2].real
    tidal_tensor[2, 2, ...] = tidal_tensor_components[5].real
    tidal_tensor[1, 2, ...] = tidal_tensor_components[4].real
    tidal_tensor[2, 1, ...] = tidal_tensor_components[4].real

    if calculate_potential:
      return tidal_tensor, potential
    
    else:
      return tidal_tensor

#-------------------------------------------------------------------------------------------------------------#

def make_traceless(matrix: np.ndarray) -> np.ndarray:
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

#-------------------------------------------------------------------------------------------------------------#

def calculate_traceless_tidal_shear(tidal_tensor: np.ndarray, grid_size: int) -> np.ndarray:
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
    traceless = make_traceless(tid)
    return traceless.reshape(grid_size, grid_size, grid_size, 3, 3)

#-------------------------------------------------------------------------------------------------------------#

def save_data(data: np.ndarray, file_path: str) -> None:
    """
    Save the given data to the specified file path.

    Parameters:
    - data: numpy.ndarray
      The data to be saved.
    - file_path: str
      The file path where the data will be saved.
    """

    try:
      np.save(file_path, data)

    except ValueError:
      print("Error saving the data to the file path.")

#-------------------------------------------------------------------------------------------------------------#

def load_all_npy_files(folder_path: str) -> list:
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

#-------------------------------------------------------------------------------------------------------------#

def calculate_eigenvalues_and_vectors(tidal_shear_tensor: np.ndarray, grid_size: int) -> tuple:
    """
    Calculate eigenvalues and eigenvectors of the tidal shear tensor for each grid point.

    Parameters:
    - tidal_shear_tensor (ndarray): 3D array representing the tidal shear tensor with shape (grid_size, grid_size, grid_size, 3, 3).
    - grid_size (int): Size of the grid in each dimension.

    Returns:
    - eigenvalues (ndarray): Array containing the eigenvalues for each grid point with shape (grid_size, grid_size, grid_size, 3).
    - eigenvectors (ndarray): Array containing the corresponding eigenvectors for each grid point with shape (grid_size, grid_size, grid_size, 3, 3).
    """

    # Reshape the tidal shear tensor to have a shape of (grid_size*grid_size*grid_size, 3, 3)
    reshaped_tensor = tidal_shear_tensor.reshape((-1, 3, 3))

    # Calculate eigenvalues and eigenvectors for all tensors simultaneously
    eigenvalues, eigenvectors = np.linalg.eig(reshaped_tensor)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues, axis=-1)[:, ::-1]

    # Apply sorting to eigenvalues and eigenvectors arrays
    eigenvalues = np.take_along_axis(eigenvalues, sorted_indices, axis=-1)
    eigenvectors = np.take_along_axis(eigenvectors, sorted_indices[..., np.newaxis], axis=-1)

    # Reshape eigenvalues and eigenvectors arrays back to the original shape
    eigenvalues = eigenvalues.reshape((grid_size, grid_size, grid_size, 3))
    eigenvectors = eigenvectors.reshape((grid_size, grid_size, grid_size, 3, 3))

    return eigenvalues, eigenvectors

#-------------------------------------------------------------------------------------------------------------#

def classify_structure(eigenvalues: np.ndarray) -> np.ndarray:
    ''' 
    Uses T-web classification scheme.

    Classifies each voxel as a part of Void, Sheet, Filament, or Node
    based on the eigenvalues of the tidal tensor.

    This counts the number of positive eigenvalues for each voxel.

    - If all three eigenvalues are positive, the voxel is classified as a Node.
    - If two eigenvalues are positive, the voxel is classified as a Filament.
    - If one eigenvalue is positive, the voxel is classified as a Sheet.
    - If no eigenvalues are positive, the voxel is classified as a Void.
    
    '''
    
    # Count the number of positive eigenvalues
    classified_array = np.sum(eigenvalues > 0, axis = -1) # Shape of eigenvalues is (grid_size, grid_size, grid_size, 3)

    return classified_array # Shape of classified_array is (grid_size, grid_size, grid_size)

#-------------------------------------------------------------------------------------------------------------#

def calculate_volume_fraction(classification_matrix: np.ndarray, label: int) -> float:
    """
    Calculate the volume fraction of a --specific label-- in a 3D classification matrix.

    Parameters:
    - classification_matrix (numpy.ndarray): 3D array containing voxel labels.
    - label (int): The label for which the volume fraction is calculated.

    Returns:
    float: The volume fraction of the specified label in the matrix.
    """
    return np.sum(classification_matrix.flatten() == label) / np.prod(classification_matrix.shape)

#-------------------------------------------------------------------------------------------------------------#

def calculate_volume_fractions(classification_matrix: np.ndarray, num_labels: np.ndarray) -> np.ndarray:
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

#-------------------------------------------------------------------------------------------------------------#

def slice_density_field(density_field: np.ndarray, slice_index: int, dim: str, slice_thickness: int) -> np.ndarray:
    """
    Extracts a 2D slice (XY plane) from a 3D density field and applies logarithmic transformation.

    Parameters:
    - density_field (numpy.ndarray): 3D array representing the density field.
    - slice_index (int): Index of the slice to be extracted along the z-axis.
    - dim (str): The dimension of the slice ('xy', 'yz', 'zx').
    - slice_thickness (int): The thickness of the slice.

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
    N = slice_thickness

    delplot1 = np.log10(density_field + 1 + eps)

    if dim == ('yz' or 'zy'):
      slic1=np.mean(delplot1[N:slice+N, :, :],axis=0)

    if dim == ('xz' or 'zx'):
      slic1=np.mean(delplot1[:, slice-N:slice+N, :],axis=1)
    
    if dim == ('xy' or 'yx'):
      slic1=np.mean(delplot1[:, :, slice-N:slice+N],axis=2)

    d_field = np.mean(delplot1[:, :, N:slice_index+N], axis = 2)
    
    return d_field

#-------------------------------------------------------------------------------------------------------------#

def get_pos(structure: str, projection: str, slice_index: int, classification: np.ndarray, grid_size: int, box_size: int) -> np.ndarray:
    
    """
    Extracts positions of points in a 2D slice based on the specified structure, projection, and classification.

    Parameters:
    - structure (str): A string representing the structure type ('v', 's', 'f', 'n').
    - projection (str): A string representing the projection type ('xy', 'yx', 'yz', 'zy', 'zx', 'xz').
    - slice_index (int): Index of the slice to be extracted.
    - classification (numpy.ndarray): 3D array representing the classification of points.
    - grid_size (int): Size of the grid in the 3D array.
    - box_size (int): Size of the box in physical units.

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
            raise ValueError('Environment not found!')
            
            
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
            raise ValueError('Environment not found!')
            
            
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
            raise ValueError('Environment not found!')
            
            
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
            raise ValueError('Environment not found!')
        
    else:
        raise ValueError('Environment not found!')
        
#-------------------------------------------------------------------------------------------------------------#

def get_class_env_pos(classification: np.ndarray, slice_index: int, dim: str, grid_size: int, box_size: int) -> list:
    
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
    - Positions are returned in physical units, scaled based on the default grid size and box size.

    Example:
    >>> classification = np.random.randint(0, 4, size=(512, 512, 512))
    >>> slice_index = 2
    >>> result = get_class_env_pos(classification, slice_index)
    """
    
    envs = ['v', 's', 'f', 'n']
    
    all_env_pos = []
    
    for env in envs:
        env_pos = get_pos(env, dim, slice_index, classification, grid_size, box_size)
        all_env_pos.append(env_pos)
    
    return all_env_pos



#----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#