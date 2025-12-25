from .No_Bypass import calculate_R_c, no_by_pass, get_R_from_pods, binary_search_feasibility
from gurobipy import Model, GRB, quicksum
from gurobipy import GRB
import numpy as np
from .No_Bypass import calculate_R_c, no_by_pass
from .Structure import init_structure
from .Route import RouteTool
from .utils import dict_to_numpy
import time
import pandas as pd


def calculate_bandwidth_utilization(N, f, d_wave, n_matrix, s_matrix):
    """
    Compute the bandwidth utilization matrix.

    Args:
        N (int): Number of nodes.
        f (np.ndarray): (N, N, N) array of splitting ratios.
        d_wave (np.ndarray): (N, N) traffic matrix.
        n_matrix (List[List[int]]): (N, N) topology matrix.
        s_matrix (np.ndarray): (N, N) capacity matrix.

    Returns:
        np.ndarray: (N, N) bandwidth utilization matrix.

    Raises:
        None.

    """
    utilization = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                numerator = sum(f[i, j_prime, j] * d_wave[i, j_prime] for j_prime in range(N)) + \
                            sum(f[i_prime, j, i] * d_wave[i_prime, j] for i_prime in range(N))

                denominator = n_matrix[i][j] * s_matrix[i, j]

                if denominator == 0:
                    if numerator != 0:
                        print("Wrong")
                        return 99999999
                    utilization[i, j] = 0
                else:
                    utilization[i, j] = numerator / denominator

    return utilization


def get_d(s_matrix, R_c, d_wave, N, tag="convex"):
    """
    Solve for traffic assignment given `s_matrix`, `R_c`, `d_wave`, and `tag`.

    Args:
        s_matrix (np.ndarray): (N, N) capacity matrix.
        R_c (np.ndarray): (N,) array of max degree per node.
        d_wave (np.ndarray): (N, N) traffic matrix.
        N (int): Number of nodes.
        tag (str, optional): "convex" for linear relaxation, otherwise solve integer/nonconvex.

    Returns:
        Union[np.ndarray, None]: (N, N) traffic allocation `D_value` if optimal; else None.

    """
    # Build model
    m = Model("NetworkOptimization")

    # Variables
    a = m.addVars(N, N, name="a")  # Direct traffic
    b = m.addVars(N, N, name="b")  # First-hop bypass
    c = m.addVars(N, N, name="c")  # Second-hop bypass

    # Direct traffic cannot exceed demand
    m.addConstrs((a[i, j] <= d_wave[i, j] for i in range(N) for j in range(N)), "FlowUpperBound")

    # Decision variables and objective
    if tag == "convex":
        x = m.addVars(N, N, name="x")  # Connection decisions
        m.setObjective(quicksum(x[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
    else:
        x = m.addVars(N, N, name="x", vtype=GRB.INTEGER)  # Connection decisions
        u = m.addVar(name="u")  # Objective variable
        m.addConstrs(
            (a[i, j] + b[i, j] + c[i, j] <= u * x[i, j] * s_matrix[i, j] for i in range(N) for j in range(N) if i != j),
            "FlowConstraintNonConvex"
        )
        m.setObjective(u, GRB.MINIMIZE)
        m.setParam('NonConvex', 2)
        m.setParam(GRB.Param.TimeLimit, 60)

    # Degree constraints
    for i in range(N):
        m.addConstr(quicksum(x[i, j] for j in range(N) if j != i) <= R_c[i], name=f"R_limit_{i}")

    # Symmetry constraints
    for i in range(N):
        for j in range(N):
            m.addConstr(x[i, j] == x[j, i], name=f"Symmetry_{i}_{j}")

    # Capacity constraints
    for i in range(N):
        for j in range(N):
            if i != j:
                m.addConstr(a[i, j] + b[i, j] + c[i, j] <= x[i, j] * s_matrix[i, j], name=f"Cost_{i}_{j}")

    # Flow balance
    for i in range(N):
        m.addConstr(
            quicksum(a[i, k] + b[i, k] for k in range(N) if k != i) ==
            quicksum(d_wave[i, j] for j in range(N) if j != i),
            name=f"FlowBalanceOut_{i}"
        )

    for j in range(N):
        m.addConstr(
            quicksum(a[k, j] + c[k, j] for k in range(N) if k != j) ==
            quicksum(d_wave[i, j] for i in range(N) if i != j),
            name=f"FlowBalanceIn_{j}"
        )

    for k in range(N):
        m.addConstr(
            quicksum(b[i, k] for i in range(N) if i != k) ==
            quicksum(c[k, j] for j in range(N) if j != k),
            name=f"FlowBalanceRelay_{k}"
        )

    # Solve
    m.optimize()

    # Results
    if m.status == GRB.OPTIMAL:
        D_value = np.zeros((N, N))
        for key in a:
            D_value[key[0]][key[1]] = a[key].x + b[key].x + c[key].x
        return D_value
    elif m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("model.ilp")
        print("No feasible solution found. Infeasible constraints are written to 'model.ilp'")
        return None
    else:
        print("No optimal solution found")
        return None


def check_u_feasibility(s_matrix, R_c, d_wave, N, u, pods):
    """
    Check feasibility for a given utilization `u`.

    Args:
        s_matrix (np.ndarray): (N, N) capacity matrix.
        R_c (List[float]): (N,) max degree per node.
        d_wave (np.ndarray): (N, N) traffic matrix.
        N (int): Number of nodes.
        u (float): Utilization candidate.
        pods (List[Pod]): Pods for obtaining degree limits.

    Returns:
        Tuple[int, np.ndarray]: (1, routing matrix) if feasible; otherwise (0, 0).

    """
    from gurobipy import Model, GRB, quicksum
    m = Model("NetworkOptimization")
    # Variables
    a = m.addVars(N, N, name="n")  # Direct traffic
    b = m.addVars(N, N, name="n")  # First-hop bypass
    c = m.addVars(N, N, name="n")  # Second-hop bypass
    m.addConstrs(a[i, j] <= d_wave[i, j] for i in range(N) for j in range(N))

    # Decision variables and objective

    x = m.addVars(N, N, name="n", vtype=GRB.INTEGER)  # Connection decisions
    # x = m.addVars(N, N, name="n")  # Connection decisions
    m.addConstrs(
        a[i, j] + b[i, j] + c[i, j] <= u * x[i, j] * s_matrix[i, j] for i in range(N) for j in range(N) if i != j)
    m.setObjective(0, GRB.MINIMIZE)
    # Degree constraints
    for i in range(N):
        m.addConstr(quicksum(x[i, j] for j in range(N) if j != i) <= R_c[i], name=f"R_limit_{i}")
    for i in range(N):
        for j in range(N):
            m.addConstr(x[i, j] == x[j, i], name=f"R_equal_{i}{j}")

    # Capacity constraints
    for i in range(N):
        for j in range(N):
            if i != j:
                m.addConstr(a[i, j] + b[i, j] + c[i, j] <= x[i, j] * s_matrix[i, j])
    for i in range(N):
        m.addConstr(quicksum([a[i, k] + b[i, k] for k in range(N) if k != i]) == quicksum(
            [d_wave[i, j] for j in range(N) if j != i]))
    for j in range(N):
        m.addConstr(quicksum([a[k, j] + c[k, j] for k in range(N) if k != j]) == quicksum(
            [d_wave[i, j] for i in range(N) if i != j]))
    for k in range(N):
        m.addConstr(quicksum([b[i, k] for i in range(N) if i != k]) == quicksum(
            [c[k, j] for j in range(N) if j != k]))
    # Set objective (just to set up the model, not optimizing)
    m.setObjective(0, GRB.MINIMIZE)

    # Solve the model
    m.setParam('OutputFlag', 1)
    m.optimize()

    # Check feasibility
    if m.status == GRB.OPTIMAL:

        D_value = np.zeros((N, N))
        for key in a:
            D_value[key[0]][key[1]] = a[key].x + b[key].x + c[key].x
        return 1, D_value
    else:
        return 0, 0


def get_optimal_u(N, d_wave, pods, s_matrix, e=0.01):
    """
    Find the optimal utilization `u` via binary search.

    Args:
        N (int): Number of nodes.
        d_wave (np.ndarray): (N, N) traffic matrix.
        pods (List[Pod]): Pods with `R` attribute for max degree.
        s_matrix (np.ndarray): (N, N) capacity matrix.
        e (float, optional): Binary search precision. Default 0.01.

    Returns:
        Tuple[float, float]: Optimal utilization and corresponding `D_estimate`.

    """
    R = get_R_from_pods(pods)
    low, high = 0, 1
    result = check_u_feasibility(s_matrix, R, d_wave, N, high, pods)
    while not result[0]:
        low = high
        high += 1

    optimal_u, D_estimate = binary_search_feasibility(low, high, s_matrix, R, d_wave, N, pods, e, check_u_feasibility)
    return optimal_u, D_estimate


def gurobi_direct_solve(s_matrix, d_wave, N, R, time):
    """
    Solve the optimization problem directly with Gurobi.

    Args:
        s_matrix (np.ndarray): (N, N) capacity matrix.
        d_wave (np.ndarray): (N, N) traffic matrix.
        N (int): Number of nodes.
        R (List[int]): (N,) max degree per node.
        time (int): Solver time limit in seconds.

    Returns:
        Tuple[Dict[Tuple[int, int, int], float], float, np.ndarray, np.ndarray]:
            - f_d: Flow allocation solution keyed by (i, j, k).
            - u_value: Utilization variable solution.
            - D_value: (N, N) traffic allocation totals.
            - n_matrix_value: (N, N) connection decision matrix.

    Raises:
        None.

    """
    from gurobipy import Model, GRB, quicksum
    f_d = {}
    # Build model
    m = Model("NetworkOptimization")
    # m.setParam('Method', 1)
    # Variables
    f = m.addVars(N, N, N, name="f", lb=0)  # Flow allocation
    n_matrix = m.addVars(N, N, vtype=GRB.INTEGER, name="n")  # Connection decisions
    u = m.addVar()  # Utilization variable
    m.setParam('NonConvex', 2)
    # Objective
    m.setObjective(u, GRB.MINIMIZE)
    # Constraints
    for i in range(N):
        m.addConstr(quicksum(n_matrix[i, j] for j in range(N) if j != i) <= R[i], name=f"R_limit_{i}")
    for i in range(N):
        for j in range(N):
            m.addConstr(n_matrix[i, j] == n_matrix[j, i], name=f"R_equal_{i}{j}")
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i == j or i == k:
                    m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")

    # Flow allocation sum constraints
    for i in range(N):
        for j in range(N):
            if i != j:
                m.addConstr(quicksum(f[i, j, k] for k in range(N)) == 1, name=f"flow_sum_{i}_{j}")

    # Capacity constraints
    for i in range(N):
        for j in range(N):
            if i != j:
                m.addConstr(quicksum(f[i, jp, j] * d_wave[i, jp] for jp in range(N)) +
                            quicksum(f[ip, j, i] * d_wave[ip, j] for ip in range(N)) <= u * n_matrix[i, j] * s_matrix[
                                i, j],
                            name=f"cost_{i}_{j}")

    # Solve model
    m.setParam('OutputFlag', 0)
    m.setParam(GRB.Param.TimeLimit, time)
    m.optimize()
    # Results
    if m.status == GRB.OPTIMAL or m.status == 9:
        for key, var in f.items():
            if isinstance(var, int) or isinstance(var, float):
                f_d[key] = var
            else:
                f_d[key] = var.x

        D_value = np.zeros((N, N))
        n_matrix_value = np.zeros((N, N))
        for key in d_wave:
            D_value[key[0]][key[1]] = sum(f_d[i, jp, j] * d_wave[i, jp] for jp in range(N)) + sum(
                f_d[ip, j, i] * d_wave[ip, j] for ip in range(N))
            n_matrix_value[key] = n_matrix[key].x

        return f_d, u.x, D_value, n_matrix_value
    else:
        print("No optimal solution found")
        pass


def SVSA(N, pods, d_wave, s_matrix, e, use_integer=False, binary=None, method="gp"):
    """
    Compute maximum utilization using the SVSA algorithm.

    Args:
        N (int): Number of nodes.
        pods (List[Pod]): Pods with `R` attribute for max degree.
        d_wave (np.ndarray): (N, N) traffic matrix.
        s_matrix (np.ndarray): (N, N) capacity matrix.
        e (float): Error tolerance.
        use_integer (bool, optional): Whether to solve with integer variables directly.
        binary (int, optional): If 1, estimate via binary search; otherwise nonconvex solve.
        method (int, optional): LP solve method; "gp" for gurobi, "rapid" for fast solver.

    Returns:
        Tuple[float, List[List[int]], np.ndarray, np.ndarray]:
        - max_u: Maximum utilization.
        - n_no_by_pass: Direct routing matrix.
        - D_estimate: Estimated traffic matrix.
        - f: (N, N, N) splitting ratio matrix.

    """
    R = [pod.R for pod in pods]
    if use_integer:
        if binary == 1:
            u_estimate, D_estimate = get_optimal_u(N, d_wave, pods, s_matrix, e=0.01)
            print(u_estimate)
        else:
            D_estimate = get_d(s_matrix, R, d_wave, N, "unconvex")
        u_real, n_no_by_pass, path = no_by_pass(N, pods, D_estimate, s_matrix, e)
        # Debug: utilization using exact traffic estimate
    else:
        D_estimate = get_d(s_matrix, R, d_wave, N, "convex")
        u_real, n_no_by_pass, path = no_by_pass(N, pods, D_estimate, s_matrix, e)
        # Debug: utilization using relaxed traffic estimate
    if method == "gp":
        f, u_now = RouteTool.lp_by_gp(n_no_by_pass, s_matrix, d_wave, N)
        f = dict_to_numpy(f, N)
    else:
        f, u_now = RouteTool.lp_rapid(np.ones(n_no_by_pass.shape), n_no_by_pass*s_matrix, d_wave, N, 200, tol=1e-6)
    utilization = calculate_bandwidth_utilization(N, f, d_wave, n_no_by_pass, s_matrix)
    max_u = np.max(utilization)
    return max_u, n_no_by_pass, D_estimate, f


def variable_separation(N, pods, d_wave, s_matrix, e, T_tmp=None, f=None, method="gp", output_flag=False):
    """
    Solve via variable separation, returning utilization, topology, and iteration traces.

    Args:
        N (int): Number of nodes.
        pods (List[Pod]): Pods with `R` attribute for max degree.
        d_wave (np.ndarray): (N, N) traffic matrix.
        s_matrix (np.ndarray): (N, N) capacity matrix.
        e (float): Precision parameter.
        T_tmp (np.ndarray, optional): Initial traffic estimate.
        f (np.ndarray, optional): Splitting ratio matrix.
        method (str, optional): "gp" or "rapid" for LP solve method.
        output_flag (bool, optional): Whether to print iteration progress.

    Returns:
        Tuple[float, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
        - u_tmp: Maximum utilization.
        - n_matrix: Final topology matrix.
        - utilization_records: Utilization per iteration.
        - n_matrix_records: Topology per iteration.
        - f: Final splitting ratios.

    """
    R = [pod.R for pod in pods]
    if T_tmp is None:
        T_tmp = d_wave
        f = RouteTool.initialize_f(N, n_matrix=np.ones((N, N)))
    else:
        f = RouteTool.initialize_f(N, n_matrix=T_tmp)
    continue_flag = True
    u_tmp = float("inf")
    i = 1
    utilization_records = []  # Track utilization per iteration
    n_matrix_records = []  # Track topology per iteration

    u_no_by_pass, n_matrix, path = no_by_pass(N, pods, T_tmp, s_matrix, e)

    while continue_flag:
        u_no_by_pass, n_matrix, path = no_by_pass(N, pods, T_tmp, s_matrix, e)
        f = dict_to_numpy(f, N)
        utilization = RouteTool.calculate_bandwidth_utilization(N, f, d_wave, n_matrix, s_matrix)
        utilization_records.append(utilization.copy())  # Record utilization for this iteration
        R_c = calculate_R_c(n_matrix, pods, N)
        RouteTool.update_n_matrix_based_on_utilization(N, R, R_c, n_matrix, utilization)
        if method == "gp":
            f, u_now = RouteTool.lp_by_gp(n_matrix, s_matrix, d_wave, N)
            f = dict_to_numpy(f, N)
        else:
            f, u_now = RouteTool.lp_rapid(np.ones(n_matrix.shape), n_matrix*s_matrix, d_wave, N, 100, tol=e)
        utilization = RouteTool.calculate_bandwidth_utilization(N, f, d_wave, n_matrix, s_matrix)
        utilization_records.append(utilization.copy())  # Record utilization for this iteration
        if u_tmp - u_now < 1e-7:
            continue_flag = False
            if output_flag:
                print(f"Final iteration {i}, current max utilization {u_tmp}")
        else:
            T_tmp = RouteTool.get_traffic(N, f, d_wave)
            u_tmp = u_now
            if output_flag:
                print(f"Iteration {i}, current max utilization {u_tmp}")
            i += 1
        n_matrix_records.append(n_matrix.copy())  # Record topology for this iteration
    return u_tmp, n_matrix, utilization_records, n_matrix_records, f


if __name__ == '__main__':
    pod_count = 40
    up_link_port_range = [256, 256]
    up_link_bandwidth_range = [100, 100]
    traffic_range = [0, 800]
    error_tolerance = 1e-4
    pods, traffic_matrix, bandwidth_matrix = init_structure(pod_count, up_link_port_range, up_link_bandwidth_range,
                                                            traffic_range, 37)
    results = []
    utilization_data = {}
    n_matrix_data = {}
    gurobi_tag = False

    methods = ["rapid","gp" ]

    for method in methods:
        print(f"\nTesting with method: {method.upper()}")

        # SVSA method
        start_time = time.time()
        max_bandwidth_svsa, svsa_matrix, traffic_estimate, f_svsa = SVSA(pod_count, pods, traffic_matrix, bandwidth_matrix,
                                                                         error_tolerance, use_integer=0, binary=1, method=method)
        time_svsa = time.time() - start_time
        utilization_svsa = calculate_bandwidth_utilization(pod_count, f_svsa, traffic_matrix, svsa_matrix, bandwidth_matrix)
        results.append([f'SVSA ({method.upper()})', max_bandwidth_svsa, time_svsa])
        utilization_data[f'SVSA_{method.upper()}'] = utilization_svsa
        n_matrix_data[f'SVSA_{method.upper()}'] = svsa_matrix.flatten()
        print(f"SVSA ({method.upper()}) Method: Max Bandwidth {max_bandwidth_svsa}, Time {time_svsa}")

        # RVSA method
        start_time = time.time()
        max_bandwidth_rvsa, rvsa_matrix, _, _, f_rvsa = variable_separation(pod_count, pods, traffic_matrix,
                                                                            bandwidth_matrix, error_tolerance,
                                                                            T_tmp=traffic_estimate, method=method)
        time_RVSA = time.time() - start_time
        utilization_RVSA = calculate_bandwidth_utilization(pod_count, f_rvsa, traffic_matrix, rvsa_matrix, bandwidth_matrix)
        results.append([f'RVSA ({method.upper()})', max_bandwidth_rvsa, time_RVSA])
        utilization_data[f'RVSA_{method.upper()}'] = utilization_RVSA
        n_matrix_data[f'RVSA_{method.upper()}'] = rvsa_matrix.flatten()
        print(f"RVSA ({method.upper()}) Method: Max Bandwidth {max_bandwidth_rvsa}, Time {time_RVSA}")

        # VSA method
        start_time = time.time()
        max_bandwidth_vsa, vsa_matrix, _, _, f_vsa = variable_separation(pod_count, pods, traffic_matrix, bandwidth_matrix,
                                                                         error_tolerance, method=method)
        time_vsa = time.time() - start_time
        utilization_vsa = calculate_bandwidth_utilization(pod_count, f_vsa, traffic_matrix, vsa_matrix, bandwidth_matrix)
        results.append([f'VSA ({method.upper()})', max_bandwidth_vsa, time_vsa])
        utilization_data[f'VSA_{method.upper()}'] = utilization_vsa
        n_matrix_data[f'VSA_{method.upper()}'] = vsa_matrix.flatten()
        print(f"VSA ({method.upper()}) Method: Max Bandwidth {max_bandwidth_vsa}, Time {time_vsa}")

    # Gurobi method (remains the same as it does not have a "method" parameter)
    if gurobi_tag:
        start_time = time.time()
        R = get_R_from_pods(pods)
        f_gurobi, max_bandwidth_gurobi, _, gurobi_matrix = gurobi_direct_solve(bandwidth_matrix, traffic_matrix, pod_count,
                                                                               R, 1000)
        time_gurobi = time.time() - start_time
        utilization_gurobi = calculate_bandwidth_utilization(pod_count, f_gurobi, traffic_matrix, gurobi_matrix,
                                                             bandwidth_matrix)
        results.append(['GUROBI', max_bandwidth_gurobi, time_gurobi])
        utilization_data['GUROBI'] = utilization_gurobi
        n_matrix_data['GUROBI'] = gurobi_matrix.flatten()
        print(f"gurobi Method: Max Bandwidth {max_bandwidth_gurobi}, Time {time_gurobi}")

    # Create a DataFrame for results
    results_df = pd.DataFrame(results, columns=['Method', 'Max Bandwidth Utilization', 'Solving Time (seconds)'])
    print(results_df)

    # Save the results to a CSV file
    results_df.to_csv('bandwidth_utilization_comparison.csv', index=False)

