import math
import numpy as np
from .Structure import init_structure
import time
from .utils import dict_to_numpy, convert_to_dict
import networkx as nx

def gp_programing(n, s_matrix, R, d_wave, N):
    """
    Solve the linear program with Gurobi to obtain the maximum bandwidth utilization and routing matrix.

    Args:
        n (int): Number of nodes.
        s_matrix (np.ndarray): (n, n) bandwidth matrix between nodes.
        R (np.ndarray): (n,) array of max degree per node.
        d_wave (np.ndarray): (n, n) traffic matrix.
        N (int): Number of nodes.

    Returns:
        Tuple[float, np.ndarray]: (max utilization, routing matrix of shape (N, N)).

    """
    from gurobipy import Model, GRB, quicksum
    # Initialize model
    m = Model("NetworkOptimization")
    # Variables
    # n = m.addVars(N, N, vtype=GRB.INTEGER, name="n")  # Connection decision variable
    n = m.addVars(N, N, name="n", vtype=GRB.INTEGER)  # Connection decision variable
    # Use v = 1/u as the objective variable to maximize v
    v = m.addVar(name="v")  # v = 1/u
    # Maximize v
    m.setObjective(v, GRB.MAXIMIZE)
    # Constraints
    # Connection degree limits
    for i in range(N):
        m.addConstr(quicksum(n[i, j] for j in range(N) if j != i) <= R[i], name=f"R_limit_{i}")
    # Symmetry constraints
    for i in range(N):
        for j in range(N):
            m.addConstr(n[i, j] == n[j, i], name=f"R_equal_{i}{j}")
    # Capacity constraints in linear form
    for i in range(N):
        for j in range(N):
            if i != j:
                # Linearized constraint: d_wave[i, j] * v <= n[i, j] * s_matrix[i, j]
                m.addConstr(d_wave[i, j] * v <= n[i, j] * s_matrix[i, j], name=f"cost_{i}_{j}")
    # Solve model
    m.setParam('OutputFlag', 0)
    m.optimize()
    # Extract results
    # if m.status == GRB.OPTIMAL:
    #     print(f"Optimal value of u: {1/v.X}")
    #     import pdb;pdb.set_trace()
    # else:
    #     print("No optimal solution found")
    u_star = {}
    for i in range(N):
        for j in range(N):
            if i != j and n[i, j].x != 0:
                u_star[(i, j)] = d_wave[i, j] / (n[i, j].x * s_matrix[i, j])
    u_star_sort = sorted(u_star.keys(), key=lambda k: u_star[k], reverse=True)
    key_max = u_star_sort[0]
    value_max = u_star[key_max]
    n_matrix = {}
    for key in n:
        n_matrix[key] = n[key].x
    return value_max, dict_to_numpy(n_matrix, N)

def gp_programing_nl(n, s_matrix, R, d_wave, N):
# def gp_programing(n, s_matrix, R, d_wave, N):
    """
    Solve the network optimization problem with Gurobi to get maximum bandwidth utilization and routing matrix.

    Args:
        n (int): Number of nodes.
        s_matrix (np.ndarray): (n, n) bandwidth matrix.
        R (np.ndarray): (n,) array of max degree per node.
        d_wave (np.ndarray): (n, n) traffic matrix.
        N (int): Number of nodes.

    Returns:
        Tuple[float, np.ndarray]: (max bandwidth utilization, routing matrix of shape (N, N)).

    """
    from gurobipy import Model, GRB, quicksum
    # Initialize model
    m = Model("NetworkOptimization")
    # Variables
    # n = m.addVars(N, N, vtype=GRB.INTEGER, name="n")  # Connection decision variable
    n = m.addVars(N, N, name="n", vtype=GRB.INTEGER)  # Connection decision variable
    u = m.addVar(name="u")  # Objective variable
    # Objective
    m.setObjective(u, GRB.MINIMIZE)
    # Constraints
    # Connection degree limits
    for i in range(N):
        m.addConstr(quicksum(n[i, j] for j in range(N) if j != i) <= R[i], name=f"R_limit_{i}")
    # Symmetry constraints
    for i in range(N):
        for j in range(N):
            m.addConstr(n[i, j] == n[j, i], name=f"R_equal_{i}{j}")
    # Capacity constraints
    for i in range(N):
        for j in range(N):
            if i != j:
                lhs = d_wave[i, j]
                rhs = u * n[i, j]*s_matrix[i,j]
                m.addConstr( 0<= rhs-lhs, name=f"cost_{i}_{j}")
    # Solve model
    m.setParam('OutputFlag', 0)
    m.optimize()
    # Extract results
    if m.status == GRB.OPTIMAL:
        print(f"Optimal value of u: {u.X}")
    else:
        print("No optimal solution found")
    u_star = {}
    for i in range(N):
        for j in range(N):
            if i != j and n[i, j].x != 0:
                u_star[(i, j)] = d_wave[i, j] / (n[i, j].x * s_matrix[i, j])
    u_star_sort = sorted(u_star.keys(), key=lambda k: u_star[k], reverse=True)
    key_max = u_star_sort[0]
    value_max = u_star[key_max]
    n_matrix = {}
    for key in n:
        n_matrix[key] = n[key].x
    return value_max, dict_to_numpy(n_matrix, N)




def is_feasible(s_matrix, R, d_wave, N, u):
    """
    Check feasibility for a given utilization u.

    Args:
        s_matrix (np.ndarray): (N, N) rate/capacity matrix.
        R (np.ndarray): (N,) array of max degree per node.
        d_wave (np.ndarray): (N, N) traffic matrix.
        N (int): Number of nodes.
        u (float): Target utilization.

    Returns:
        Tuple[int, np.ndarray, float]: (1, n_matrix, u) if feasible, otherwise (0, 0, u).

    """
    # Convert inputs to NumPy arrays for vectorized ops
    if not isinstance(d_wave, np.ndarray):
        d_wave_np = dict_to_numpy(d_wave, N)
    else:
        d_wave_np = d_wave.copy()
    
    if not isinstance(s_matrix, np.ndarray):
        s_matrix_np = dict_to_numpy(s_matrix, N)
    else:
        s_matrix_np = s_matrix.copy()
    
    # Create n_matrix as a NumPy array
    n_matrix_np = np.zeros((N, N), dtype=float)
    
    # Compute n_matrix (vectorized); mask to avoid divide-by-zero
    mask = (s_matrix_np > 0) & (np.eye(N) == 0)
    
    # Compute d_wave / (u * s_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.zeros((N, N))
        ratio[mask] = d_wave_np[mask] / (u * s_matrix_np[mask])
    
    # Take elementwise max with transpose to mirror pairs
    max_ratio = np.maximum(ratio, ratio.T)
    
    # Round up
    n_matrix_np[mask] = np.ceil(max_ratio[mask])
    
    # Convert to dict for calculate_R_c compatibility
    n_matrix = convert_to_dict(n_matrix_np)
    
    # Check degree constraints
    R_c = calculate_R_c(n_matrix, R, N)
    # import pdb;pdb.set_trace()
    if R_c[0]:
        # Recompute realized utilization u
        u_new = 0
        # Vectorized maximum utilization computation
        with np.errstate(divide='ignore', invalid='ignore'):
            utilization = np.zeros((N, N))
            utilization_mask = (n_matrix_np > 0) & (s_matrix_np > 0)
            utilization[utilization_mask] = d_wave_np[utilization_mask] / (n_matrix_np[utilization_mask] * s_matrix_np[utilization_mask])
            u_new = np.max(utilization)
        
        return 1, n_matrix_np, u_new
    else:
        return 0, 0, u


def update_u_sort_list(u_sort_list, U_matrix, updated_key):
    """
    Update `u_sort_list` ordering after a value change.

    Args:
    - u_sort_list (List[Tuple[int, int]]): Current sorted list of node pairs (i, j) by utilization.
    - U_matrix (Dict[Tuple[int, int], float]): Map of (i, j) to utilization values.
    - updated_key (Tuple[int, int]): Node pair (i, j) whose value changed.

    Returns:
    - List[Tuple[int, int]]: Updated sorted utilization list.

    """
    # Drop previous value
    u_sort_list = [key for key in u_sort_list if key != updated_key]
    # Find insertion index
    updated_value = U_matrix[updated_key]
    insert_index = 0
    for i, key in enumerate(u_sort_list):
        if U_matrix[key] < updated_value:
            insert_index = i
            break
    else:
        insert_index = len(u_sort_list)
    # Insert updated value
    u_sort_list.insert(insert_index, updated_key)
    return u_sort_list


def calculate_R_c(n_matrix, R, N):
    """
    Compute realized degree `R_c` for each node.

    Args:
        n_matrix (Union[Dict[Tuple[int, int], int], np.ndarray]): Topology matrix as dict or NumPy array.
        pods (List[Pod]): Pods with `R` field for maximum degree.
        N (int): Node count.

    Returns:
        Tuple[List[int], Union[int, str]]: On success, list of realized degrees and None; on violation, (0, "fail").

    Raises:
        ValueError: If `n_matrix` is neither a dict nor NumPy array.

    """
    R_c = []
    for i in range(N):
        if isinstance(n_matrix, dict):
            # Use dictionary access and sum for dict-based matrix
            tmp = sum(n_matrix.get((i, j), 0) for j in range(N))
        elif isinstance(n_matrix, np.ndarray):
            # Use NumPy sum for array-based matrix
            tmp = np.sum(n_matrix[i, :])
        else:
            # import pdb;pdb.set_trace()
            raise ValueError("Unsupported matrix type")
        # import pdb;pdb.set_trace()
        if tmp > R[i]:
            return 0, "fail"
        R_c.append(tmp)
    
    return R_c


def binary_search_feasibility(low, high, s_matrix, R, d_wave, N, e, check_feasibility_func):
    """
    Binary search for the maximal feasible utilization value.

    Args:
        low (float): Lower bound for search.
        high (float): Upper bound for search.
        s_matrix (np.ndarray): (N, N) bandwidth matrix.
        R (float): Total port capacity per node.
        d_wave (np.ndarray): (N,) or (N, N) demand representation.
        N (int): Number of nodes.
        e (float): Binary search tolerance.
        check_feasibility_func (Callable): Function to test feasibility for a given utilization.

    Returns:
        Tuple[float, int]: Best feasible utilization and corresponding n_matrix.

    """
    n_matrix = 0
    timings = []
    feasible_solutions = []
    start_time = time.time()
    if abs(high - low) < e:
        high = low + 2*e
    contiue=True
    while contiue:

        mid = (low + high) / 2
        result = check_feasibility_func(s_matrix, R, d_wave, N, mid)

        if result[0]:
            high = mid
            n_matrix = result[1]
            elapsed_time = time.time() - start_time
            timings.append(elapsed_time)
            feasible_solutions.append(mid)
        else:
            low = mid
        if abs(high - low) < e:
            contiue=False
            result = check_feasibility_func(s_matrix, R, d_wave, N, high)
            n_matrix = result[1]
            elapsed_time = time.time() - start_time
            timings.append(elapsed_time)
            feasible_solutions.append(mid)
    return high, n_matrix


def get_R_from_pods(pods):
    return [pod.R for pod in pods]
import numpy as np
import networkx as nx

import numpy as np
import networkx as nx

import numpy as np
import networkx as nx

# def compute_routing(N, d_wave, s_matrix, e=0.01, k_d=2):import numpy as np
import networkx as nx

import numpy as np
import networkx as nx

def compute_routing(N, d_wave, s_matrix, e=0.01, k_d=2):
    d_wave_orig = d_wave.copy()
    d_wave = np.maximum(d_wave, d_wave.T)
    np.fill_diagonal(d_wave, 0)

    n_matrix = np.zeros((N, N), dtype=np.float64)

    for round_id in range(k_d):
        remaining_demand = np.maximum(d_wave - n_matrix * s_matrix, 0)
        P = np.zeros((N, N))

        # Stage 1: greedy vectorized matching
        demand_unconnected = (d_wave > 0) & (n_matrix == 0)
        pairs = np.argwhere(np.triu(demand_unconnected, 1))
        matched = set()
        for i, j in pairs:
            if i not in matched and j not in matched:
                P[i, j] = P[j, i] = 1
                matched.update([i, j])

        # Stage 2: Hungarian assignment for remaining unmatched nodes
        unmatched_src = [i for i in range(N) if i not in matched]
        unmatched_dst = [j for j in range(N) if j not in matched]

        if unmatched_src and unmatched_dst:
            cost_matrix = -remaining_demand[np.ix_(unmatched_src, unmatched_dst)]
            cost_matrix[remaining_demand[np.ix_(unmatched_src, unmatched_dst)] < e] = 0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_pairs = 0
            for r, c in zip(row_ind, col_ind):
                u, v = unmatched_src[r], unmatched_dst[c]
                if remaining_demand[u, v] >= e and s_matrix[u, v] > 0:
                    P[u, v] = 1
                    matched_pairs += 1

        # Stage 3: fallback greedy matching
        if np.sum(P) == 0:
            potential_edges = np.argwhere(np.triu((remaining_demand >= e) & (s_matrix > 0), 1))
            matched = set()
            for i, j in potential_edges:
                if i not in matched and j not in matched:
                    P[i, j] = P[j, i] = 1
                    matched.update([i, j])

        # Stop if no matches are found
        if np.sum(P) == 0:
            break
        else:
            n_matrix += P

    # Utilization computation
    total_demand = d_wave_orig
    total_capacity = s_matrix * (n_matrix)
    demand_mask = total_demand > 0
    connection_mask = (n_matrix + n_matrix.T) > 0
    
    # Exclude diagonal elements when checking connectivity
    diagonal_mask = ~np.eye(N, dtype=bool)
    if np.any((demand_mask & ~connection_mask & diagonal_mask)):
        import pdb;pdb.set_trace()
        return None, float('inf')
    
    utilization = np.zeros_like(d_wave)
    with np.errstate(divide='ignore', invalid='ignore'):
        utilization[connection_mask] = total_demand[connection_mask] / total_capacity[connection_mask]

    u_max_1 = np.max(utilization) if np.any(utilization) else 0.0
    # import pdb;pdb.set_trace()
    return n_matrix, u_max_1

import numpy as np
import networkx as nx

def compute_routing_solstice_discrete_updated(d_wave, s_matrix, k_d=2):
    """
    Simplified Solstice-style scheduling:
    - Each round finds a perfect matching (no thresholding)
    - After applying P, update demand T = max(T - S * P, 0)
    - Run for k_d rounds

    Args:
        d_wave: NxN traffic matrix
        s_matrix: NxN link capacity matrix
        k_d: number of scheduling rounds (switches)

    Returns:
        n_matrix: NxN scheduling matrix
        u_max: maximum link utilization
        utilization: utilization matrix
    """
    N = d_wave.shape[0]
    n_matrix = np.zeros((N, N), dtype=np.float64)
    T = d_wave.copy().astype(np.float64)  # Remaining demand matrix

    rounds = 0
    while rounds < k_d:
        # Build bipartite graph for positive demand entries
        G = nx.Graph()
        left = [f"s{i}" for i in range(N)]
        right = [f"r{j}" for j in range(N)]
        G.add_nodes_from(left, bipartite=0)
        G.add_nodes_from(right, bipartite=1)

        for i in range(N):
            for j in range(N):
                if i != j and T[i, j] > 0:
                    G.add_edge(f"s{i}", f"r{j}")

        matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=left)
        if len(matching) // 2 < N:
            break  # Cannot schedule further

        # Build permutation matrix
        P = np.zeros((N, N))
        for i in range(N):
            matched = matching.get(f"s{i}")
            if matched:
                j = int(matched[1:])
                P[i, j] = 1

        # Update connection matrix
        n_matrix += P

        # Update demand: T = max(T - S * P, 0)
        T = np.maximum(T - s_matrix * P, 0)

        rounds += 1

    # Compute maximum link utilization
    total_demand = d_wave + d_wave.T
    total_capacity = s_matrix * n_matrix
    utilization = np.zeros_like(d_wave)
    with np.errstate(divide='ignore', invalid='ignore'):
        utilization[n_matrix > 0] = total_demand[n_matrix > 0] / total_capacity[n_matrix > 0]
    u_max = np.max(utilization) if np.any(utilization) else 0.0

    return n_matrix, u_max, utilization

def get_initial_solution_no_by_pass(N, d_wave, R, s_matrix, e=0.01):
    """
    Obtain an initial solution without bypass.

    Args:
        N (int): Number of nodes.
        d_wave (float): Traffic matrix.
        s_matrix (np.ndarray): Bandwidth matrix.
        e (float, optional): Precision. Defaults to 0.01.

    Returns:
        Tuple[np.ndarray, float]: (routing matrix n_matrix, optimal_u).

    """
    # R = get_R_from_pods(pods)
    low, high = 0, 1
    result = is_feasible(s_matrix, R, d_wave, N, high)
    while not result[0]:
        low = high
        high += 1
        # import pdb;pdb.set_trace()
        result = is_feasible(s_matrix, R, d_wave, N, high)


    optimal_u, n_matrix = binary_search_feasibility(low, high, s_matrix, R, d_wave, N, e, is_feasible)
    if np.sum(n_matrix) == 0:
        import pdb;pdb.set_trace()
    return n_matrix, optimal_u


import numpy as np
from scipy.optimize import linear_sum_assignment

def bvn_decompose_optimized(T, tol=1e-6, max_iter=100):
    """
    Perform a Birkhoff-von Neumann decomposition on a doubly stochastic matrix T.
    Returns a list of (P, λ) pairs where P is a permutation matrix and λ its weight.
    """
    T = T.copy()
    N = T.shape[0]
    decomposition = []

    for _ in range(max_iter):
        T[T < tol] = 0
        if np.max(T) <= tol:
            break

        cost = -T  # Negate to perform maximum-weight matching
        row_ind, col_ind = linear_sum_assignment(cost)
        P = np.zeros_like(T)
        for i, j in zip(row_ind, col_ind):
            P[i, j] = 1

        lambda_val = np.min(T[P == 1])
        T[P == 1] -= lambda_val
        decomposition.append((P, lambda_val))

    return decomposition

def construct_n_matrix(decomposition, k_d=None, symmetric=True):
    """
    Build n_matrix from a BVN decomposition, optionally using only the first k_d permutations.
    """
    N = decomposition[0][0].shape[0]
    n_matrix = np.zeros((N, N), dtype=np.float64)

    count = 0
    for P, _ in decomposition:
        if k_d is not None and count >= k_d:
            break
        if symmetric:
            n_matrix += P
        else:
            n_matrix += P
        count += 1

    return n_matrix

def calculate_utilization(d_wave, s_matrix, n_matrix):
    """
    Compute maximum link utilization.
    """
    total_demand = d_wave + d_wave.T
    total_capacity = s_matrix * n_matrix
    utilization = np.zeros_like(d_wave)
    with np.errstate(divide='ignore', invalid='ignore'):
        utilization[n_matrix > 0] = total_demand[n_matrix > 0] / total_capacity[n_matrix > 0]
    u_max = np.max(utilization) if np.any(utilization) else 0.0
    return u_max, utilization

def compute_routing_bvn(d_wave, s_matrix, k_d=2, tol=1e-6):
    """
    Schedule via BVN decomposition using the first k_d permutations.

    Args:
        d_wave: (N x N) traffic demand matrix.
        s_matrix: (N x N) link capacity matrix.
        k_d: Max scheduling rounds (number of switches).
        tol: Decomposition tolerance.

    Returns:
        n_matrix: (N x N) scheduling matrix.
        u_max: Maximum link utilization.
        utilization: (N x N) utilization matrix.
    """
    # Normalize traffic matrix (approximate doubly stochastic)
    T = d_wave.copy().astype(np.float64)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_norm = T / row_sums

    # Decompose
    decomposition = bvn_decompose_optimized(T_norm, tol=tol)

    # Build schedule matrix using the first k_d matchings
    n_matrix = construct_n_matrix(decomposition, k_d=k_d, symmetric=True)
    # Compute utilization
    u_max, utilization = calculate_utilization(d_wave, s_matrix, n_matrix)

    return n_matrix, u_max, utilization


def no_by_pass(N, R, d_wave, s_matrix, e):
    # Directly find optimal non-bypass routing
    n_matrix, _ = get_initial_solution_no_by_pass(N, d_wave, R, s_matrix, e)
    # Feasibility check and adjustments
    if isinstance(n_matrix, int) :
        return None,None
    R_c = calculate_R_c(n_matrix, R, N)
    U_matrix = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                U_matrix[(i, j)] = 0
            else:
                if n_matrix[(i, j)] == 0:
                    U_matrix[(i, j)] = 0
                else:
                    U_matrix[(i, j)] = d_wave[(i, j)] / (n_matrix[(i, j)] * s_matrix[(i, j)])
    u_sort_list = sorted(U_matrix.keys(), key=lambda k: U_matrix[k], reverse=True)
    u_max_1 = U_matrix[u_sort_list[0]]
    u_max_2 = U_matrix[u_sort_list[1]]
    i_1 = u_sort_list[0][0]
    j_1 = u_sort_list[0][1]
    n_plus = min(R[i_1] - R_c[i_1], R[j_1] - R_c[j_1],
                 max(math.ceil(d_wave[(i_1, j_1)] / (s_matrix[(i_1, j_1)] * u_max_2) - n_matrix[(i_1, j_1)]), 1))
    # print(n_plus)
    while n_plus > 0:
        # print(n_plus)
        R_c = calculate_R_c(n_matrix, R, N)
        n_plus = min(R[i_1] - R_c[i_1], R[j_1] - R_c[j_1],
                     max(math.ceil(d_wave[(i_1, j_1)] / (s_matrix[(i_1, j_1)] * u_max_2) - n_matrix[(i_1, j_1)]), 1))
        if n_plus > 0:
            # Update separately to avoid write-order issues
            n_matrix[(i_1, j_1)] += n_plus
            U_matrix[(i_1, j_1)] = d_wave[(i_1, j_1)] / (n_matrix[(i_1, j_1)] * s_matrix[(i_1, j_1)])
            if U_matrix[(i_1, j_1)] > u_max_2:
                return n_matrix
            else:
                u_sort_list = update_u_sort_list(u_sort_list, U_matrix, (i_1, j_1))
                n_matrix[(j_1, i_1)] += n_plus
                U_matrix[(j_1, i_1)] = d_wave[(j_1, i_1)] / (n_matrix[(j_1, i_1)] * s_matrix[(j_1, i_1)])
                u_sort_list = update_u_sort_list(u_sort_list, U_matrix, (j_1, i_1))
                u_max_1 = U_matrix[u_sort_list[0]]
                u_max_2 = U_matrix[u_sort_list[1]]
                i_1 = u_sort_list[0][0]
                j_1 = u_sort_list[0][1]
    return u_max_1, n_matrix, u_sort_list[0]


if __name__ == '__main__':
    pod_count = 30
    up_link_port_range = [2560, 2560]
    up_link_bandwidth_range = [100, 100]
    traffic_range = [0, 1000]
    error_tolerance = 1e-12

    pods, traffic_matrix, bandwidth_matrix = init_structure(pod_count, up_link_port_range, up_link_bandwidth_range,
                                                            traffic_range, 20)

    # Optimization without bypass traffic splitting
    R = [pod.R for pod in pods]
    start_time = time.time()
    max_bandwidth_direct, direct_matrix, _ = no_by_pass(pod_count, pods, traffic_matrix, bandwidth_matrix,
                                                        error_tolerance)
    time_direct = time.time() - start_time
    print(
        f"Proposed method: max utilization {max_bandwidth_direct}, link count {np.sum(direct_matrix)}, solve time {time.time() - start_time}")

    start_time = time.time()
    u_no_by_pass_gp, n_no_by_pass_gp = gp_programing(pod_count, bandwidth_matrix, R, traffic_matrix, pod_count)
    print(
        f"Mixed-integer program: max utilization {u_no_by_pass_gp}, link count {np.sum(n_no_by_pass_gp)}, solve time {time.time() - start_time}")
    start_time = time.time()
    # a = calculate_R_c(n_no_by_pass_gp, pods, N)
    print(1)
