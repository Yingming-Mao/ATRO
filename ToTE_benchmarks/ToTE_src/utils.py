import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# Removed deprecated commented plotting helper for clarity


def convert_to_dict(matrix):
    """
    Convert a NumPy matrix to a dict mapping (i, j) -> value.

    Args:
        matrix (np.ndarray): The NumPy matrix to convert.

    Returns:
        Union[Dict[Tuple[int, int], float], Any]: The converted dict; if the input is not a NumPy array, returns input as-is.

    """
    # Check input type is ndarray
    if isinstance(matrix, np.ndarray):
        # Initialize an empty dict
        matrix_dict = {}
        # Iterate over all elements in the matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # Store the index tuple and corresponding value
                matrix_dict[(i, j)] = matrix[i, j]
        return matrix_dict
    else:
        # Return input as-is if not a NumPy matrix
        return matrix


def dict_to_numpy(matrix_dict, N):
    """
    Convert a dict-form matrix to a NumPy matrix.

    Args:
        matrix_dict (dict or np.ndarray): Dict-form matrix or NumPy array.
        N (int): Matrix dimension.

    Returns:
        np.ndarray: Converted NumPy matrix.

    """
    if isinstance(matrix_dict, np.ndarray):
        numpy_matrix = matrix_dict
    else:
        dim = len(next(iter(matrix_dict.keys())))
        if dim != 2:
            numpy_matrix = np.zeros((N, N, N))
            # Iterate through dict and fill 3D matrix
            for key, value in matrix_dict.items():
                i, j, k = key  # Keys are tuples (i, j, k)
                numpy_matrix[i, j, k] = value
        else:
            # Create an N x N zero matrix
            numpy_matrix = np.zeros((N, N))
            # Iterate through dict and fill 2D matrix
            for key, value in matrix_dict.items():
                i, j = key  # Keys are tuples (i, j)
                numpy_matrix[i, j] = value

    return numpy_matrix


def create_submatrices(original_matrix, n):
    original_matrix = np.array(original_matrix)

    # Get indices of non-zero elements in the original matrix
    non_zero_indices = np.argwhere(original_matrix != 0)

    # If fewer than n non-zero elements exist, raise an error
    if len(non_zero_indices) < n:
        raise ValueError("Not enough non-zero elements to create the required submatrices.")

    # Shuffle the non-zero indices randomly
    np.random.shuffle(non_zero_indices)

    # Split indices into n groups
    groups = np.array_split(non_zero_indices, n)

    # Initialize list of submatrices
    submatrices = []
    shape = original_matrix.shape
    # Assign each group to a separate submatrix
    for group in groups:
        submatrix = np.zeros(shape, dtype=original_matrix.dtype)
        for row, col in group:
            submatrix[row, col] = original_matrix[row, col]
        submatrices.append(submatrix)

    return submatrices


# Sorting by the "traffic", here represented by the sum of non-zero elements in each submatrix
def sort_submatrices_by_traffic(submatrices):
    return sorted(submatrices, key=lambda x: np.sum(x), reverse=True)


# Split the largest submatrix based on the algorithm
def split_largest_matrix(submatrices, split_ratio=0.5):
    largest_matrix = submatrices.pop(0)  # Pop the largest matrix (already sorted)
    non_zero_indices = np.argwhere(largest_matrix != 0)

    if len(non_zero_indices) <= 1:
        # If there is only one non-zero element or none, we can't split further
        return submatrices

    # Randomly shuffle the indices to split
    np.random.shuffle(non_zero_indices)

    # Split into two halves
    half = int(len(non_zero_indices) * split_ratio)
    group1, group2 = non_zero_indices[:half], non_zero_indices[half:]

    # Create two new matrices
    shape = largest_matrix.shape
    submatrix1 = np.zeros(shape, dtype=largest_matrix.dtype)
    submatrix2 = np.zeros(shape, dtype=largest_matrix.dtype)

    for row, col in group1:
        submatrix1[row, col] = largest_matrix[row, col]

    for row, col in group2:
        submatrix2[row, col] = largest_matrix[row, col]

    # Add the two new matrices back to the list
    submatrices.extend([submatrix1, submatrix2])

    return submatrices


def split_until_limit(submatrices, t, n):
    # Initialize a queue (simulating a max heap based on traffic/size)
    queue = sort_submatrices_by_traffic(submatrices)  # Assumes sorted initially by traffic (size)

    # While queue length is less than or equal to (1 + t) * n, continue splitting
    while len(queue) <= (1 + t) * n:
        # Split the largest matrix in the queue
        queue = split_largest_matrix(queue)

    return queue



import numpy as np
import os
import glob
from collections import defaultdict
import sys

import scipy
# from sklearn.cluster import KMeans

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import DATA_DIR


def Get_peak_demand(dm_list):
    """Get the peak demand from the demand matrix."""
    dm_matrixs = np.array(dm_list)
    predict_dm = np.max(dm_matrixs, axis=0)
    return predict_dm


def Get_edge_to_path(topology, candidate_path):
    """ Get the mapping from edge to path."""
    if candidate_path == None:
        return None
    edge_to_path = {}
    for edge in topology.edges:
        edge_to_path[(int(edge[0]), int(edge[1]))] = []
    for src in topology.nodes:
        for dst in topology.nodes:
            if src != dst:
                for index, path in enumerate(candidate_path[(src, dst)]):
                    for i in range(len(path) - 1):
                        edge_to_path[(int(path[i]), int(path[i + 1]))].append((int(src), int(dst), index))
    return edge_to_path


def linear_get_dir(props, is_test):
    """Get the train or test directory for the given topology."""
    postfix = "test" if is_test else "train"
    return os.path.join(DATA_DIR, props.topo_name, postfix)


def linear_get_hists_from_folder(folder):
    """Get the list of histogram files from the given folder."""
    hists = sorted(glob.glob(folder + "/*.hist"))
    return hists


def paths_from_file(paths_file, num_nodes):
    """Get the paths from the file."""
    pij = defaultdict(list)
    pid = 0
    with open(paths_file, 'r') as f:
        lines = sorted(f.readlines())
        lines_dict = {line.split(":")[0]: line for line in lines if line.strip() != ""}
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst:
                    continue
                try:
                    if "%d %d" % (src, dst) in lines_dict:
                        line = lines_dict["%d %d" % (src, dst)].strip()
                    else:
                        line = [l for l in lines if l.startswith("%d %d:" % (src, dst))]
                        if line == []:
                            continue
                        line = line[0]
                        line = line.strip()
                    if not line: continue
                    i, j = list(map(int, line.split(":")[0].split(" ")))
                    paths = line.split(":")[1].split(",")
                    for p_ in paths:
                        node_list = list(map(int, p_.split("-")))
                        pij[(i, j)].append(node_list)
                        pid += 1
                except Exception as e:
                    print(e)
                    import pdb;
                    pdb.set_trace()
    return pij


def Get_common_cases_tms(hist_tms):
    """Get the common cases traffic demands from the history traffic demands.

    Args:
        hist_tms: the history traffic demands.
    """
    # Computing the convex hull can be challenging when the length of hist_tms is particularly large,
    # or when the dimensionality of each tm is very high.
    # Therefore, we use all hist_tms as common_cases_tms.

    # hull = ConvexHull(hist_tms)
    # hull_vertices = hull.vertices
    # common_case_tms = [hist_tms[i] for i in hull_vertices]
    common_case_tms = hist_tms
    return common_case_tms


def restore_flattened_to_original(flattened_f, original_shape, mask_3d):
    # Create an empty array with the original shape to restore data
    restored_f = np.zeros(original_shape)

    # Fill flattened data back into the original array using the mask
    if isinstance(flattened_f,np.matrix):
        restored_f[mask_3d]=flattened_f.A.flatten()
    elif isinstance(flattened_f,scipy.sparse._coo.coo_matrix):
        restored_f[mask_3d] = flattened_f.toarray().flatten()
    else:
        restored_f[mask_3d] = flattened_f.flatten()


    return restored_f


def dict_to_numpy_3d(path_weights):
    """
    Convert a dict with keys 'w_i_j_k' to a 3D NumPy array; fill unspecified entries with zeros.

    Args:
        path_weights (dict): Dict with keys 'w_i_j_k' and float values.

    Returns:
        np.ndarray: A 3D NumPy array with zeros for unspecified entries.
    """
    # Extract max indices i, j, k to initialize the array
    max_i = max(int(key.split('_')[1]) for key in path_weights.keys()) + 1
    max_j = max(int(key.split('_')[2]) for key in path_weights.keys()) + 1
    max_k = max(int(key.split('_')[3]) for key in path_weights.keys()) + 1

    # Initialize 3D array with zeros
    array = np.zeros((max_i, max_j, max_k))

    # Fill the array
    for key, value in path_weights.items():
        _, i, j, k = key.split('_')
        i, j, k = int(i), int(j), int(k)
        array[i, j, k] = value

    return array


def numpy_3d_to_dict(array, topology, candidate_path):
    """
    Convert a 3D NumPy array to a dict with keys 'w_i_j_k'.
    Stores all values (can filter zeros externally if desired).

    Args:
        array (np.ndarray): 3D NumPy array.
        topology: Topology object providing number of nodes.
        candidate_path (dict): Candidate paths for each node pair (i, j).

    Returns:
        dict: Dict keyed by 'w_i_j_k' with float values.
    """
    path_weights = {}

    # Iterate over node pairs and path indices
    for i in range(topology.number_of_nodes()):
        for j in range(topology.number_of_nodes()):
            if j != i:
                num_paths = len(candidate_path[(i, j)])  # Number of paths from i to j
                for k in range(num_paths):
                    # if array[i, j, k] != 0:  # Store only non-zero entries if desired
                    key = f'w_{i}_{j}_{k}'
                    path_weights[key] = array[i, j, k]

    return path_weights

def dict_to_numpy(matrix_dict, N):
    """
    Convert a dict-form matrix to a NumPy matrix.

    Args:
        matrix_dict (dict or np.ndarray): Dict form matrix or NumPy array.
        N (int): Matrix dimension.

    Returns:
        np.ndarray: Converted NumPy matrix.

    """
    if isinstance(matrix_dict, np.ndarray):
        numpy_matrix = matrix_dict
    else:
        dim = len(next(iter(matrix_dict.keys())))
        if dim != 2:
            numpy_matrix = np.zeros((N, N, N))
            # Iterate through dict and fill 3D matrix
            for key, value in matrix_dict.items():
                i, j, k = key  # Keys are tuples (i, j, k)
                numpy_matrix[i, j, k] = value
        else:
            # Create an N x N zero matrix
            numpy_matrix = np.zeros((N, N))
            # Iterate through dict and fill 2D matrix
            for key, value in matrix_dict.items():
                i, j = key  # Keys are tuples (i, j)
                numpy_matrix[i, j] = value

    return numpy_matrix

def create_submatrices(original_matrix, n):
    original_matrix = np.array(original_matrix)

    # Get indices of non-zero elements in the original matrix
    non_zero_indices = np.argwhere(original_matrix != 0)

    # If the number of non-zero elements is less than n, raise an error
    if len(non_zero_indices) < n:
        raise ValueError("Not enough non-zero elements to create the required submatrices.")

    # Randomly shuffle non-zero element indices
    np.random.shuffle(non_zero_indices)

    # Split indices into n groups
    groups = np.array_split(non_zero_indices, n)

    # Initialize submatrix list
    submatrices = []
    shape = original_matrix.shape

    # Assign a group of elements to each submatrix
    for group in groups:
        submatrix = np.zeros(shape, dtype=original_matrix.dtype)
        for row, col in group:
            submatrix[row, col] = original_matrix[row, col]
        submatrices.append(submatrix)

    return submatrices


# Sorting by the "traffic", here represented by the sum of non-zero elements in each submatrix
def sort_submatrices_by_traffic(submatrices):
    return sorted(submatrices, key=lambda x: np.sum(x), reverse=True)


# Split the largest submatrix based on the algorithm
def split_largest_matrix(submatrices, split_ratio=0.5):
    largest_matrix = submatrices.pop(0)  # Pop the largest matrix (already sorted)
    non_zero_indices = np.argwhere(largest_matrix != 0)

    if len(non_zero_indices) <= 1:
        # If there is only one non-zero element or none, we can't split further
        return submatrices

    # Randomly shuffle the indices to split
    np.random.shuffle(non_zero_indices)

    # Split into two halves
    half = int(len(non_zero_indices) * split_ratio)
    group1, group2 = non_zero_indices[:half], non_zero_indices[half:]

    # Create two new matrices
    shape = largest_matrix.shape
    submatrix1 = np.zeros(shape, dtype=largest_matrix.dtype)
    submatrix2 = np.zeros(shape, dtype=largest_matrix.dtype)

    for row, col in group1:
        submatrix1[row, col] = largest_matrix[row, col]

    for row, col in group2:
        submatrix2[row, col] = largest_matrix[row, col]

    # Add the two new matrices back to the list
    submatrices.extend([submatrix1, submatrix2])

    return submatrices


def split_until_limit(submatrices, t, n):
    # Initialize a queue (simulating a max heap based on traffic/size)
    queue = sort_submatrices_by_traffic(submatrices)  # Assumes sorted initially by traffic (size)

    # While queue length is less than or equal to (1 + t) * n, continue splitting
    while len(queue) <= (1 + t) * n:
        # Split the largest matrix in the queue
        queue = split_largest_matrix(queue)

    return queue
