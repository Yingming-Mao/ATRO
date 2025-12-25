import time
import numpy as np
try:
    import gurobipy as gp
    from gurobipy import Model, GRB, quicksum
    HAS_GUROBI = True
except Exception:
    gp = None
    Model = GRB = quicksum = None
    HAS_GUROBI = False
from ToTE_benchmarks.ToTE_src.utils import dict_to_numpy
from .No_Bypass import no_by_pass, calculate_R_c,get_initial_solution_no_by_pass,gp_programing,compute_routing,gp_programing_nl,compute_routing_bvn,compute_routing_solstice_discrete_updated
from .Route import RouteTool
from .linear_routing import Routing
from .utils import Get_peak_demand, dict_to_numpy_3d, numpy_3d_to_dict
import math
from ortools.graph.python import min_cost_flow  # OR-Tools min-cost flow
class linear_algorithm(object):

    def __init__(self, props, topo, candidate_path, edge_to_path):
        self.props = props
        self.routing = Routing(topo, candidate_path, edge_to_path,props)

    def solve_traffic_engineering(self):
        pass





class ATRO(linear_algorithm):

    def __init__(self, props, topo, candidate_path, edge_to_path):
        super(ATRO, self).__init__(props, topo, candidate_path, edge_to_path)
    def solve(self, cluster_pod_num, spine_up_port_num, T_a_b, init_T_a_b=None):
        # Validate inputs
        pod_num = cluster_pod_num
        if not isinstance(T_a_b, np.ndarray):
            T_a_b = np.array(T_a_b)
        assert T_a_b.shape == (pod_num, pod_num)
        pod_up_port_num = spine_up_port_num

        # Default init_T_a_b to zeros when missing
        if init_T_a_b is None:
            init_T_a_b = np.zeros((pod_num, pod_num))

        arc_capacity_k0_map = {}
        arc_cost_k0_map = {}
        for i in range(pod_num):
            for j in range(pod_num):
                u = T_a_b[i, j]
                c = pod_up_port_num
                if i == j:
                    arc_capacity_k0_map[f'{i}_{j}'] = [0]
                    arc_cost_k0_map[f'{i}_{j}'] = [0]
                elif u < c:
                    arc_capacity_k0_map[f'{i}_{j}'] = [u, c - u]
                    arc_cost_k0_map[f'{i}_{j}'] = [-1, 1]
                else:
                    arc_capacity_k0_map[f'{i}_{j}'] = [u]
                    arc_cost_k0_map[f'{i}_{j}'] = [-1]

        start_nodes = []  # Supply nodes: 0..pod_num-1
        end_nodes = []
        capacities = []
        unit_costs = []

        supplies_map = {}

        global_arc_id = 0
        i_j_t_localarc_2_globalarc_map = {}

        for i in range(pod_num):
            for j in range(pod_num):
                supply_node_id_k0 = i
                end_node_id_k0 = pod_num + j
                for arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}'])):
                    if arc_capacity_k0_map[f'{i}_{j}'][arc_id] > 0:
                        start_nodes.append(supply_node_id_k0)
                        end_nodes.append(end_node_id_k0)
                        capacities.append(int(arc_capacity_k0_map[f'{i}_{j}'][arc_id]))
                        unit_costs.append(arc_cost_k0_map[f'{i}_{j}'][arc_id])
                        i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{arc_id}_0'] = global_arc_id
                        global_arc_id += 1

        supplies = []
        for i in range(2 * pod_num):
            if i < pod_num:
                supplies.append(int(pod_up_port_num))
            else:
                supplies.append(-1 * int(pod_up_port_num))

        for i in range(pod_num):
            for j in range(pod_num):
                supplies[i] -= int(init_T_a_b[i, j])
                supplies[j + pod_num] += int(init_T_a_b[i, j])

        min_cost_flow_ = min_cost_flow.SimpleMinCostFlow()

        # Add each arc.
        for i in range(0, len(start_nodes)):
            min_cost_flow_.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i],
                                                                capacities[i], unit_costs[i])

        # Add node supplies.
        for i in range(0, len(supplies)):
            min_cost_flow_.set_node_supply(i, int(supplies[i]))

        if min_cost_flow_.solve() == min_cost_flow_.OPTIMAL:
            c_ij = np.zeros((pod_num, pod_num), dtype=int)
            for i in range(pod_num):
                for j in range(pod_num):
                    for local_arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}'])):
                        if arc_capacity_k0_map[f'{i}_{j}'][local_arc_id] > 0:
                            global_arc_id = i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{local_arc_id}_0']
                            c_ij[i, j] += min_cost_flow_.flow(global_arc_id)
                    c_ij[i, j] += int(init_T_a_b[i, j])
            return c_ij

        else:
            success_MCF = False
            assert success_MCF
            print('There was an issue with the min cost flow input.')
    def absm(self,r, d_wave, T_tmp=None, method="rapid", output_flag=False):
        peak_demand = d_wave
        s_matrix = np.ones(self.routing.topology.capacity_matrix.shape)*self.props.base_capacity
        d_wave = d_wave.reshape(s_matrix.shape)

        N = s_matrix.shape[0]
        R = [r]*N

        n_matrix, u_tmp = get_initial_solution_no_by_pass(N,  d_wave, R, s_matrix, 1e-7)
        # import pdb;pdb.set_trace()    
        return u_tmp, n_matrix

    def bg(self,r, d_wave, T_tmp=None, method="rapid", output_flag=False):
        s_matrix = np.ones(self.routing.topology.capacity_matrix.shape)*self.props.base_capacity
        d_wave = d_wave.reshape(s_matrix.shape)

        N = s_matrix.shape[0]
 
        
        T_tmp = d_wave  
        u_tmp = float("inf")

        n_matrix, u_tmp = compute_routing(N, T_tmp, s_matrix, 1e-7,r)
        return u_tmp, n_matrix
    def milp_d(self, r, d_wave, T_tmp=None, method="rapid", output_flag=False):
        s_matrix = np.ones(self.routing.topology.capacity_matrix.shape) * self.props.base_capacity
        d_wave = d_wave.reshape(s_matrix.shape)
        N = s_matrix.shape[0]
        R = [r] * N
        if T_tmp is None:
            T_tmp = d_wave
            f = RouteTool.initialize_f(N, n_matrix=np.ones((N, N)))
        else:
            f = RouteTool.initialize_f(N, n_matrix=T_tmp)
        u_tmp, n_matrix = gp_programing(N, s_matrix, R, T_tmp, N)
        return u_tmp, n_matrix

    def atro(self,r, d_wave, T_tmp=None, method="rapid", output_flag=False):

        s_matrix = np.ones(self.routing.topology.capacity_matrix.shape)*self.props.base_capacity 
        d_wave = d_wave.reshape(s_matrix.shape)
        N = s_matrix.shape[0]
        R = [r]*N
        mask_matrix = self.routing.mask_matrix
        utilization_records = []  
        T_tmp = d_wave      
        f = RouteTool.initialize_f(N, n_matrix=np.ones((N, N)))
        n_matrix, u_start = get_initial_solution_no_by_pass(N, T_tmp+1e-1, R, s_matrix, 1e-7)
        utilization_records.append(u_start)
        continue_flag = True
        u_tmp = float("inf")
        i = 1        
        n_matrix_records = []  # Record n_matrix for each iteration
        R_c = calculate_R_c(n_matrix, R, N)
        denominator = n_matrix * s_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            utilization = np.where(denominator != 0, T_tmp / denominator, 0)
        RouteTool.update_n_matrix_based_on_utilization(N, R, R_c, n_matrix, utilization,T_tmp,d_wave)
        np.fill_diagonal(n_matrix, 0)
        f = RouteTool.initialize_f(N, n_matrix=n_matrix)
        if N<=99:
            u_now,f = RouteTool.lp_by_gp(n_matrix, s_matrix, d_wave, N,mask_matrix)
        else:
            
            u_now,f = RouteTool.lp_rapid(np.ones(n_matrix.shape), n_matrix*s_matrix, d_wave.reshape(s_matrix.shape),
                                             s_matrix.shape[0], 2000, f, mask_matrix, 1000000, tol=1e-6)
        u_tmp=u_now
        utilization_records.append(u_now)
        T_tmp = RouteTool.get_traffic(N, f, d_wave)
        while continue_flag:
            f = dict_to_numpy(f, N)
            n_matrix, u_d = get_initial_solution_no_by_pass(N, T_tmp, R, s_matrix, 1e-6)
            R_c = calculate_R_c(n_matrix, R, N)
            denominator = n_matrix * s_matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                utilization = np.where(denominator != 0, T_tmp / denominator, 0)
            utilization_records.append(u_d) 
            RouteTool.update_n_matrix_based_on_utilization(N, R, R_c, n_matrix, utilization,T_tmp,d_wave)
            if method == "gp":
                u_now,f = RouteTool.lp_by_gp(n_matrix, s_matrix, d_wave, N)
                f = dict_to_numpy(f, N)
            else:

                u_now,f = RouteTool.lp_rapid(np.ones(n_matrix.shape), n_matrix*s_matrix, d_wave.reshape(s_matrix.shape),
                                             s_matrix.shape[0], 2000, f, mask_matrix, 1000000, tol=1e-8)
                # print("end")
                utilization_records.append(u_now)   
                f = dict_to_numpy(f, N)
            if u_tmp - u_now < 1e-5:
                continue_flag = False
                if output_flag:
                    print(f"{i}iterration,MLU is {u_tmp}")
            else:
                T_tmp = RouteTool.get_traffic(N, f, d_wave)
                u_tmp = u_now
                if output_flag:
                    print(f"{i}iterration,MLU is {u_tmp}")
                i += 1

            n_matrix_records.append(n_matrix.copy())
        
        return u_tmp, n_matrix, utilization_records, n_matrix_records, f

    
    def restore_to_integer(self,N, d_star, R):
        # Parameters and model setup
        model = gp.Model("Integer_Recovery")

        # Decision variables
        x = model.addVars(N, N, vtype=GRB.INTEGER, name="x")

        # Objective: maximize total integer allocations within bounds
        model.setObjective(gp.quicksum(x[i, j] for i in range(N) for j in range(N)), GRB.MAXIMIZE)

        # Constraints: bound by floor/ceil of fractional solution and enforce symmetry
        for i in range(N):
            for j in range(N):
                model.addConstr(x[i, j] >= np.floor(d_star[i, j]), name=f"lower_bound_{i}_{j}")
                model.addConstr(x[i, j] <= np.ceil(d_star[i, j]), name=f"upper_bound_{i}_{j}")
                model.addConstr(x[i, j] == x[j, i], name=f"symmetry_{i}_{j}")

        for i in range(N):
            model.addConstr(gp.quicksum(x[j, i] for j in range(N)) <= R[i], name=f"row_sum_{i}")

        # Solve
        model.setParam('OutputFlag', False)
        model.optimize()

        # Return solution
        if model.status == GRB.OPTIMAL:
            solution = model.getAttr('x', x)
            return solution


class COUDER(linear_algorithm):

    def __init__(self, props, topo, candidate_path, edge_to_path):
        super(COUDER, self).__init__(props, topo, candidate_path, edge_to_path)

    def solve(self, cluster_pod_num, spine_up_port_num, T_a_b, init_T_a_b=None):
        # Validate inputs
        pod_num = cluster_pod_num
        if not isinstance(T_a_b, np.ndarray):
            T_a_b = np.array(T_a_b)
        assert T_a_b.shape == (pod_num, pod_num)
        pod_up_port_num = spine_up_port_num

        # Default init_T_a_b to zeros when missing
        if init_T_a_b is None:
            init_T_a_b = np.zeros((pod_num, pod_num))

        arc_capacity_k0_map = {}
        arc_cost_k0_map = {}
        for i in range(pod_num):
            for j in range(pod_num):
                u = T_a_b[i, j]
                c = pod_up_port_num
                if i == j:
                    arc_capacity_k0_map[f'{i}_{j}'] = [0]
                    arc_cost_k0_map[f'{i}_{j}'] = [0]
                elif u < c:
                    arc_capacity_k0_map[f'{i}_{j}'] = [u, c - u]
                    arc_cost_k0_map[f'{i}_{j}'] = [-1, 1]
                else:
                    arc_capacity_k0_map[f'{i}_{j}'] = [u]
                    arc_cost_k0_map[f'{i}_{j}'] = [-1]

        start_nodes = []  # Supply nodes: 0..pod_num-1
        end_nodes = []
        capacities = []
        unit_costs = []

        supplies_map = {}

        global_arc_id = 0
        i_j_t_localarc_2_globalarc_map = {}

        for i in range(pod_num):
            for j in range(pod_num):
                supply_node_id_k0 = i
                end_node_id_k0 = pod_num + j
                for arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}'])):
                    if arc_capacity_k0_map[f'{i}_{j}'][arc_id] > 0:
                        start_nodes.append(supply_node_id_k0)
                        end_nodes.append(end_node_id_k0)
                        # Ensure capacity is integer
                        capacities.append(int(arc_capacity_k0_map[f'{i}_{j}'][arc_id]))
                        unit_costs.append(arc_cost_k0_map[f'{i}_{j}'][arc_id])
                        i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{arc_id}_0'] = global_arc_id
                        global_arc_id += 1

        supplies = []
        for i in range(2 * pod_num):
            if i < pod_num:
                # Ensure supplies are integer
                supplies.append(int(pod_up_port_num))
            else:
                supplies.append(-1 * int(pod_up_port_num))

        for i in range(pod_num):
            for j in range(pod_num):
                # Use integer arithmetic for adjustments
                supplies[i] -= int(init_T_a_b[i, j])
                supplies[j + pod_num] += int(init_T_a_b[i, j])

        min_cost_flow_ = min_cost_flow.SimpleMinCostFlow()

        # Add each arc.
        for i in range(0, len(start_nodes)):
            min_cost_flow_.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i],
                                                                capacities[i], unit_costs[i])

        # Add node supplies.
        for i in range(0, len(supplies)):
            # Ensure supply is integer
            min_cost_flow_.set_node_supply(i, int(supplies[i]))

        if min_cost_flow_.solve() == min_cost_flow_.OPTIMAL:
            c_ij = np.zeros((pod_num, pod_num), dtype=int)
            for i in range(pod_num):
                for j in range(pod_num):
                    for local_arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}'])):
                        if arc_capacity_k0_map[f'{i}_{j}'][local_arc_id] > 0:
                            global_arc_id = i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{local_arc_id}_0']
                            c_ij[i, j] += min_cost_flow_.flow(global_arc_id)
                    c_ij[i, j] += int(init_T_a_b[i, j])
            return c_ij

        else:
            success_MCF = False
            assert success_MCF
            print('There was an issue with the min cost flow input.')
    def mcf(self,r,d_wave, tag="Continue"):
        # Use props.base_capacity as the per-link base capacity
        s_matrix = np.ones(self.routing.topology.capacity_matrix.shape)*self.props.base_capacity
        mask_matrix = self.routing.mask_matrix
        # self.routing.topology.capacity_matrix / (self.props.base_capacity/1e9)
        N = s_matrix.shape[0]
        d_wave = d_wave.reshape(s_matrix.shape)
        R_c = [r]*N
        n_frac = d_wave/s_matrix
        # import pdb;pdb.set_trace()
        
        # Build initial topology: mark demand presence with double loop
        init_T = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                init_T[i][j] = 1 if n_frac[i][j] > 0 else 0
        
        # Apply mask_matrix
        mask_matrix = self.routing.mask_matrix
        n_matrix_mcf = self.solve(N, r,n_frac, init_T)
        # Enforce symmetry by taking min of symmetric entries
        for i in range(N):
            for j in range(i+1, N):  # Iterate upper triangular only
                min_val = min(n_matrix_mcf[i,j], n_matrix_mcf[j,i])
                n_matrix_mcf[i,j] = min_val
                n_matrix_mcf[j,i] = min_val
                # Validity check placeholder
        # if np.any((init_T == 1) & (n_value_int == 0)):
        #     n_value_int = n_matrix_mcf.copy()
        # Compute max link utilization
        # Avoid divide-by-zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            u_now = np.max(np.where(n_matrix_mcf * s_matrix > 0, 
                                   d_wave / (n_matrix_mcf * s_matrix), 
                                   0))
        
        # Return max utilization and connection matrix
        return u_now, n_matrix_mcf
    def couder(self,r,d_wave, tag="Continue"):
        # Use props.base_capacity as the per-link base capacity
        s_matrix = np.ones(self.routing.topology.capacity_matrix.shape)*self.props.base_capacity
        mask_matrix = self.routing.mask_matrix
        # self.routing.topology.capacity_matrix / (self.props.base_capacity/1e9)
        N = s_matrix.shape[0]
        d_wave = d_wave.reshape(s_matrix.shape)
        R_c = [r]*N
        # Initialize model
        m = Model("NetworkOptimization")
        # m.setParam('Seed', 4)
        # Variables
        f = m.addVars(N, N, N, name="f", lb=0)  # Flow allocation variables
        m.setParam('OutputFlag', False)
        if tag == "INTEGER":
            n_matrix = m.addVars(N, N, vtype=GRB.INTEGER, name="n")  # Connection decision variables
        else:
            n_matrix = m.addVars(N, N, name="n")  # Connection decision variables
        # Flow allocation and connectivity constraints
        for i in range(N):
            m.addConstr(quicksum(n_matrix[i, j] for j in range(N) if j != i) <= R_c[i], name=f"R_limit_{i}")
        for i in range(N):
            for j in range(N):
                m.addConstr(n_matrix[i, j] == n_matrix[j, i], name=f"R_equal_{i}{j}")
        # Objective
        u = m.addVar()
        m.setObjective(u, GRB.MAXIMIZE)

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i == j or i == k:
                        m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")
                    if mask_matrix[i,j,k]==0:
                        m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")
        # Flow sum constraints per demand
        for i in range(N):
            for j in range(N):
                if i != j:
                    m.addConstr(quicksum(f[i, j, k] for k in range(N)) == u, name=f"flow_sum_{i}_{j}")
        # Capacity constraints
        for i in range(N):
            for j in range(N):
                if i != j:
                    m.addConstr(quicksum(f[i, jp, j] * d_wave[i, jp] for jp in range(N)) +
                                quicksum(f[ip, j, i] * d_wave[ip, j] for ip in range(N)) <= n_matrix[i, j] * s_matrix[
                                    i, j],
                                name=f"cost_{i}_{j}")
        # Solve model
        # m.setParam('TimeLimit', 10000)
        m.setParam('OutputFlag', False)
        m.optimize()
        f_d = np.zeros((N, N, N))
        for key in f:
            f_d[key] = f[key].x 
        f_d=f_d/u.x
        
        # Compute flow per edge
        edge_flows = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Flow on edge (i, j)
                    flow_ij = 0
                    # From i via j to others
                    for jp in range(N):
                        flow_ij += f_d[i, jp, j] * d_wave[i, jp]
                    # From others via i to j
                    for ip in range(N):
                        flow_ij += f_d[ip, j, i] * d_wave[ip, j]
                    edge_flows[i, j] = flow_ij
        
        # Use min-cost flow to compute connection counts

        
        # Return results
        if m.status == GRB.OPTIMAL:
            if tag == "INTEGER":
                n_value = np.zeros((N, N))
                for key in n_matrix:
                    n_value[key[0]][key[1]] = n_matrix[key].x
                # import pdb;pdb.set_trace()
                utilization = RouteTool.calculate_bandwidth_utilization(N, f_d, d_wave, n_value, s_matrix)                
                return 1 / u.x, f_d, n_value,
            else:
                n_value = np.zeros((N, N))
                for key in n_matrix:
                    n_value[key[0]][key[1]] = n_matrix[key].x
                init_T = np.zeros((N, N))
                for key in n_matrix:
                    init_T[key[0]][key[1]] = 1 if n_matrix[key].x > 0 else 0

                solution = self.restore_to_integer(N, n_value,n_matrix, R_c,init_T)
                
                n_value_int = np.zeros((N, N))
                for key in n_matrix:
                    n_value_int[key[0]][key[1]] = solution[key]
                
                mask_matrix = self.routing.mask_matrix
                n_matrix_mcf = self.solve(N, r,n_value, init_T)
                for i in range(N):
                    for j in range(i+1, N):  # Iterate upper triangular only
                        min_val = min(n_matrix_mcf[i,j], n_matrix_mcf[j,i])
                        n_matrix_mcf[i,j] = min_val
                        n_matrix_mcf[j,i] = min_val
                # import pdb;pdb.set_trace()
                u_now, f = RouteTool.lp_by_gp(n_value_int, s_matrix, d_wave, N, mask_matrix)
                if np.all(f==100):
                    # import pdb;pdb.set_trace()
                    u_now, f = RouteTool.lp_by_gp(n_matrix_mcf, s_matrix, d_wave, N, mask_matrix)
                # utilization = RouteTool.calculate_bandwidth_utilization(N, f, d_wave, n_value_int, s_matrix)
                # import pdb;pdb.set_trace()
                return u_now, f, n_value_int

        else:
            print("No optimal solution found")
            return 0, 0, np.array(0)

    def restore_to_integer(self,N, d_star,n_matrix, R,init_T):
        # Parameters and model setup
        model = gp.Model("Integer_Recovery")

        # Decision variables
        x = model.addVars(N, N, vtype=GRB.INTEGER, name="x")

        # Objective: maximize total integer allocations within bounds
        model.setObjective(gp.quicksum(x[i, j] for i in range(N) for j in range(N)), GRB.MAXIMIZE)

        # Constraints: bound by floor/ceil of fractional solution and enforce symmetry
        for i in range(N):
            for j in range(N):
                
                model.addConstr(x[i, j] >= np.floor(d_star[i, j]), name=f"lower_bound_{i}_{j}")
                model.addConstr(x[i, j] <= np.ceil(d_star[i, j]), name=f"upper_bound_{i}_{j}")
                model.addConstr(x[i, j] == x[j, i], name=f"symmetry_{i}_{j}")
                # import pdb;pdb.set_trace()
                # if init_T[i,j]>0:
                #     if i!=j:
                #         model.addConstr(x[i, j] >= 1, name=f"symmetry_{i}_{j}")


        for i in range(N):
            model.addConstr(gp.quicksum(x[j, i] for j in range(N)) <= R[i], name=f"row_sum_{i}")

        # Solve
        model.setParam('OutputFlag', False)
        model.optimize()
        
        # Return solution
        if model.status == GRB.OPTIMAL:
            solution = model.getAttr('x', x)
            # import pdb;pdb.set_trace()
            return solution
    def to_integer_topo_in_COUDER(self, d_star, R):
        """
        Map fractional topology to integer topology (similar to to_integer_topo).
        :param d_star: Fractional topology matrix
        :param R: Resource limits per node
        :return: Integer topology matrix
        """
        ori_shape = d_star.shape

        # Flatten to list
        flat_topo = d_star.flatten().tolist()

        # Record fractional part for each element
        index_frac_dict = {}
        for i in range(len(flat_topo)):
            index_frac_dict[i] = math.modf(flat_topo[i])[0]

        # Sort by fractional part in descending order
        index_frac_list = sorted(index_frac_dict.items(), key = lambda x : x[1], reverse = True)

        # Floor to get initial integer topology
        integer_topo = np.floor(d_star)

        # Compute egress/ingress sums per node
        egress_floor_sum = np.sum(integer_topo, axis=1)
        ingress_floor_sum = np.sum(integer_topo, axis=0)
        res_r_egress = R - egress_floor_sum
        res_r_ingress = R - ingress_floor_sum

        # Iterate by fractional part order and increment where resources allow
        for (index, _) in index_frac_list:
            row = math.floor(index / len(R))
            column = index % len(R)
            if row == column:
                continue
            if res_r_egress[row] > 0 and res_r_ingress[column] > 0:
                res_r_egress[row] -= 1
                res_r_ingress[column] -= 1
                integer_topo[row][column] += 1

        return integer_topo

