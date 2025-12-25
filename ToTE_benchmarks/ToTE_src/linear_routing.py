import copy
import random
from copy import deepcopy

import numpy as np
import gurobipy as grb
from collections import Counter, defaultdict
from ToTE_benchmarks.ToTE_src.utils import restore_flattened_to_original
from ToTE_benchmarks.ToTE_src.utils import Get_edge_to_path
from scipy.sparse import csr_matrix
import numpy as np
from scipy import sparse
import time
from collections import defaultdict
class Routing:
    def __init__(self, topology, candidate_path, edge_to_path,props):
        """Initialize the routing with the topology, candidate paths and edge.

        Args:
            topology: the network topology
            candidate_path: the candidate paths for each s-d pair
            edge_to_path: the mapping from edge to path
        """
        self.mask = None
        self.mask_3d = None
        self.spread = None
        self.f = None
        self.topology = topology
        self.candidate_path = candidate_path
        self.edge_to_path = edge_to_path
        # self.path_capacity = self.Get_path_capacity()
        self.mask_matrix = self.calculate_mask_matrix()
        
        

        # Construct SDtoPath and PathtoEdge matrices
        if props.type == "sparse":
            if props.TE_solver in ["SSDO","SSDO_hot","ATRO"]:
                self.PathtoEdge, self.SD_pair_to_row, self.edge_list,self.SDtoPath_dict,self.PathtoEdge_dict = self.construct_SDtoPath_and_PathtoEdge()
            
            N = self.topology.number_of_nodes()

            # Initialize 2D boolean mask for edges present in topology
            mask = np.zeros((N, N), dtype=bool)
            for i in range(N):
                for j in range(N):
                    if self.topology.has_edge(i, j):
                        mask[i, j] = True

            # Save mask for later restoration
            self.uti_restore = mask

            # Filter invalid edges in self.C based on mask
            self.C = self.topology.capacity_matrix[mask]
            # self.C_r = np.reciprocal(self.C)
            a = []
            for i in self.C:
                a.append(1/i)
            self.C_r = np.array(a)
        else:
            # self.mask_matrix = self.calculate_mask_matrix()
            pass
            

        # Flatten self.C if needed
        C_flattened = self.topology.capacity_matrix.flatten()
        

            
    def get_MLU(self,demands,u_now1):
        util=[]
        N=int(np.sqrt(demands.size))
        demands.resize((N,N))
        for edge in self.topology.edges:
            util.append(sum(
                            u_now1[f'w_{src}_{dst}_{k}'] * demands[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                        )/self.topology.edges[edge]['capacity'])
        
        mlu1=max(util)
        return mlu1

    def calculate_mask_matrix(self):
        N = self.topology.number_of_nodes()
        if self.candidate_path is None:
            # If candidate_path is empty, set mask_matrix to all ones
            return np.ones((N, N, N), dtype=float)
        # Otherwise compute mask_matrix
        mask_matrix = np.zeros((N, N, N), dtype=float)
        for (i, j), paths in self.candidate_path.items():
            for path in paths:
                if len(path) == 2:
                    mask_matrix[i, j, j] = 1
                if len(path) >= 4:
                    continue
                for k in path[1:-1]:  # intermediate nodes
                    mask_matrix[i, j, k] = 1
        return mask_matrix

    def renew(self):
        """
        Generate all possible paths for each (i, j) pair, including direct paths and one-hop paths through an intermediate node.

        Args:
            topology: An object that contains the network topology, specifically the adjacency information.

        Returns:
            valid_paths: A defaultdict containing all valid paths for each (i, j) pair.
        """
        N = self.topology.number_of_nodes()
        n_matrix = self.topology.capacity_matrix()
        self.candidate_path = defaultdict(list)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Add direct path
                if n_matrix[i, j] != 0:
                    self.candidate_path[(i, j)].append([i, j])

                # Add one-hop path
                for k in range(N):
                    if k != i and k != j:
                        if n_matrix[i, k] != 0 and n_matrix[k, j] != 0:
                            self.candidate_path[(i, j)].append([i, k, j])
        self.path_capacity = self.Get_path_capacity()
        self.edge_to_path = Get_edge_to_path(self.topology, self.candidate_path)

    def MLU_traffic_engineering(self, demands):
        """Compute the traffic engineering solutions for multiple demands to minimize the worst-case MLU.

        Args:
            demands: the traffic demands, shape: (demand_number, number_of_nodes * number_of_nodes)
        """
        if self.candidate_path == None:
            self.renew()
        for i in range(len(demands)):
            demands[i] = demands[i].reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0
        mlu = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i in range(self.topology.number_of_nodes())
                            for j in range(self.topology.number_of_nodes())
                            if j != i
                            for k in range(len(self.candidate_path[(i, j)]))
                            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        # the sum of the routing weights of the candidate paths for each s-d pair should be 1
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )

        for demand in demands:
            m.addConstrs(
                grb.quicksum(
                    path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                ) <= mlu * self.topology.edges[edge]['capacity']
                for edge in self.topology.edges
            )

        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')

    def rapid_programming_path(self, demand, spread, epoch, tol=1e-7,init_ratio=None,time_limit=100000):
        N = self.topology.number_of_nodes()
        time1=time.time()
        # Initialize path split ratio matrix f
        if init_ratio is None:
            self.f = self.initialize_flow_split_matrix_with_numpy()  # f[i, j, k] denotes fraction of pair (i, j) on the k-th path
        else:
            self.f=init_ratio
        self.calculate_flatten_mask()
        self.spread = spread

        # Ensure demand shape is (N, N)
        if demand.shape != (N, N):
            demand = demand.reshape((N, N))

        bandwidth_matrix = self.topology.capacity_matrix
        traffic_matrix = demand

        # Compute initial bandwidth utilization
        utilization,utilization_1 = self.calculate_bandwidth_utilization_path(traffic_matrix)
        utilization[bandwidth_matrix == 0] = 0
        # Get mapping from (i, j) to SDtoPath row index
        SD_pair_to_row = self.SD_pair_to_row
        start_time=time.time()
        opt = np.max(utilization_1)
        change_tag=1
        max_utilization_history = []
        time_history = []
        for count in range(epoch):
            
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            max_indices = np.argwhere(rounded_utilization == max_value)
            # pairs = self.get_start_end_array(max_indices)
            pairs = self.get_communication_pairs_for_high_bandwidth_edges(max_indices)

            def sort_key(pair):
                i, j = pair
                return demand[i, j]
            # pairs = sorted(pairs, key=sort_key, reverse=True)
            # print(1)
            for i, j in pairs:
                time_1=time.time()
                # Use SD_pair_to_row to get row index for (i, j)
                row_idx = SD_pair_to_row[(i, j)]  # Get row index from mapping
                # SDtoPath_row = self.SDtoPath[row_idx, :]  # Obtain row for (i, j)
                
                now_path_list=self.SDtoPath_dict[row_idx]
                if len(now_path_list)==1 :

                    continue
                    # print(1)

                f_slice = self.f[self.mask_3d]

                flow_on_path_now=f_slice[now_path_list]*traffic_matrix[i, j]

                

                # if i==2 and j==3:
                #     print(1)
                
                # Step 4: Map path flow to edge flow (using PathtoEdge sparse matrix)
                for index,iss in enumerate(now_path_list):
                    edge_list=self.PathtoEdge_dict[iss]
                    # print(utilization_1[edge_list] - flow_on_path_now[index] * self.C_r[edge_list])
                    utilization_1[edge_list] = utilization_1[edge_list] - flow_on_path_now[index] * self.C_r[edge_list]
                    
                  
                
                # Remove deprecated commented alternative implementation for clarity
                
                time_2=time.time()
                # Find maximum utilization
                u_max_prime = np.max(utilization_1)

                utilization[self.uti_restore] = utilization_1
                # Update split ratios f_ij
                if demand[i, j] == 0:
                    f_ij = np.zeros(len(self.candidate_path[(i, j)]))
                else:
                    f_ij = self.update_f_path(utilization, u_max_prime, i, j, demand)
                
                # Ensure split ratios sum to 1
                if f_ij.sum() >= 1:
                    optimal_u = self.binary_search_optimal_u(utilization, self.update_f_path, i, j, demand, tol=tol)
                    if demand[i, j] == 0:
                        f_ij = np.zeros(len(self.candidate_path[(i, j)]))
                    else:
                        f_ij = self.update_f_path(utilization, optimal_u, i, j, demand)

                    if f_ij.sum() == 0:
                        f_ij = np.zeros(len(f_ij))
                    else:
                        f_ij = f_ij / f_ij.sum()  # Normalize
                    self.f[i, j, :len(f_ij)] = f_ij

                    f_slice = self.f[self.mask_3d]

                    
                    for index,iis in enumerate(now_path_list) :
                        edge_list=self.PathtoEdge_dict[iis]
                        # print(flow_on_path_now[index],self.C[edge_list])

                        flow_on_path_now=f_slice[now_path_list]*traffic_matrix[i, j]
                        utilization_1[edge_list]=utilization_1[edge_list]+flow_on_path_now[index]*self.C_r[edge_list]
                    utilization[self.uti_restore] = utilization_1

                    
        
                else:
                    # Binary search optimal split ratio and adjust
                    optimal_u = self.binary_search_optimal_u(utilization, self.update_f_path, i, j, demand,
                                                             upper_bound=opt,
                                                             lower_bound=u_max_prime, target_sum=1, tol=tol)
                    if demand[i, j] == 0:
                        f_ij = np.zeros(len(self.candidate_path[(i, j)]))
                    else:
                        f_ij = self.update_f_path(utilization, optimal_u, i, j, demand)

                    if f_ij.sum() == 0:
                        f_ij = np.zeros(len(f_ij))
                    else:
                        f_ij = f_ij / f_ij.sum()  # Normalize
                    self.f[i, j, :len(f_ij)] = f_ij
                    f_slice = self.f[self.mask_3d]

                    
                    for index,iis in enumerate(now_path_list) :
                        edge_list=self.PathtoEdge_dict[iis]

                        flow_on_path_now=f_slice[now_path_list]*traffic_matrix[i, j]
                        utilization_1[edge_list]=utilization_1[edge_list]+flow_on_path_now[index]*self.C_r[edge_list]
                        # print(flow_on_path_now[index]/self.C[edge_list])
                    utilization[self.uti_restore] = utilization_1
                time_3=time.time()
       
            # Check convergence
            if opt - np.max(utilization) <= tol * 10:
                break
                if change_tag==0:
                    change_tag = 1
                else:
                    break
            else:
                if time_3-time1>time_limit:
                    break

            # Update optimal utilization
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)


        # Final recomputation of utilization
        utilization,utilization_1 = self.calculate_bandwidth_utilization_path(traffic_matrix)

        max_value = np.max(utilization)

        max_value = np.max(utilization)
        max_indices = np.argwhere(utilization == max_value)
        # print(max_indices)
        return max_value, self.f

    def static_rapid_programming_path(self, demand, spread, epoch, tol=1e-7):
        N = self.topology.number_of_nodes()

        # Initialize path split ratio matrix f
        self.f = self.initialize_flow_split_matrix_with_numpy()  # f[i, j, k] denotes fraction of pair (i, j) on the k-th path
        self.calculate_flatten_mask()
        self.spread = spread

        # Ensure demand shape is (N, N)
        if demand.shape != (N, N):
            demand = demand.reshape((N, N))

        bandwidth_matrix = self.topology.capacity_matrix
        traffic_matrix = demand

        # Compute initial bandwidth utilization
        utilization,utilization_1 = self.calculate_bandwidth_utilization_path(traffic_matrix)
        utilization[bandwidth_matrix == 0] = 0
        # Get mapping from (i, j) to SDtoPath row index
        SD_pair_to_row = self.SD_pair_to_row
        start_time=time.time()
        opt = np.max(utilization_1)
        change_tag=1
        max_utilization_history = []
        time_history = []
        for count in range(epoch):
            
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            max_indices = np.argwhere(rounded_utilization == max_value)
            # pairs = self.get_start_end_array(max_indices)
            # Get coordinates of non-zero elements
            nonzero_coords = np.nonzero(demand)

            # Pair the coordinates
            pairs = list(zip(nonzero_coords[0], nonzero_coords[1]))

            # Remove diagonal pairs
            pairs = [(i, j) for i, j in pairs if i != j]

            def sort_key(pair):
                i, j = pair
                return demand[i, j]
            # pairs = sorted(pairs, key=sort_key, reverse=True)
            # print(1)
            for i, j in pairs:
                time_1=time.time()
                # Use SD_pair_to_row to get row index for (i, j)
                row_idx = SD_pair_to_row[(i, j)]  # from mapping dict
                # SDtoPath_row = self.SDtoPath[row_idx, :]  # Obtain row for (i, j)
                
                now_path_list=self.SDtoPath_dict[row_idx]
                if len(now_path_list)==1 :

                    continue
                    # print(1)

                # Step 3: Handle dense matrix self.f[self.mask_3d]
                f_slice = self.f[self.mask_3d]

                flow_on_path_now=f_slice[now_path_list]*traffic_matrix[i, j]
        
                
                # Step 4: Map path flow to edge flow (using PathtoEdge sparse matrix)
                for index,iss in enumerate(now_path_list):
                    edge_list=self.PathtoEdge_dict[iss]
                    # print(utilization_1[edge_list]-flow_on_path_now[index]*self.C_r[edge_list])
                    utilization_1[edge_list]=utilization_1[edge_list]-flow_on_path_now[index]*self.C_r[edge_list]
                    
  
                time_2=time.time()
                # Find maximum utilization
                u_max_prime = np.max(utilization_1)

                utilization[self.uti_restore] = utilization_1
                # Update split ratios f_ij
                if demand[i, j] == 0:
                    f_ij = np.zeros(len(self.candidate_path[(i, j)]))
                else:
                    f_ij = self.update_f_path(utilization, u_max_prime, i, j, demand)
                optimal_u = self.binary_search_optimal_u(utilization, self.update_f_path, i, j, demand,lower_bound=np.min(utilization),upper_bound=opt, target_sum=1, tol=tol)
                if demand[i, j] == 0:
                    f_ij = np.zeros(len(self.candidate_path[(i, j)]))
                else:
                    f_ij = self.update_f_path(utilization, optimal_u, i, j, demand)

                if f_ij.sum() == 0:
                    f_ij = np.zeros(len(f_ij))
                else:
                    f_ij = f_ij / f_ij.sum()  # Normalize
                self.f[i, j, :len(f_ij)] = f_ij
                f_slice = self.f[self.mask_3d]

                
                for index,iis in enumerate(now_path_list) :
                    edge_list=self.PathtoEdge_dict[iis]

                    flow_on_path_now=f_slice[now_path_list]*traffic_matrix[i, j]
                    utilization_1[edge_list]=utilization_1[edge_list]+flow_on_path_now[index]*self.C_r[edge_list]
                    # print(flow_on_path_now[index]/self.C[edge_list])
                utilization[self.uti_restore] = utilization_1
           

            # Check convergence
            if opt - np.max(utilization) <= tol * 10:
                break
                if change_tag==0:
                    change_tag = 1
                else:
                    break

            # Update optimal utilization
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)


        # Final utilization recomputation
        utilization,utilization_1 = self.calculate_bandwidth_utilization_path(traffic_matrix)

        max_value = np.max(utilization)


        return max_value, self.f,max_utilization_history,time_history

    def LP_base_rapid_programming_path(self, demand, spread, epoch, tol=1e-7):
        N = self.topology.number_of_nodes()

        # Initialize path split-ratio matrix f
        self.f = self.initialize_flow_split_matrix_with_numpy()  # f[i, j, k] denotes fraction of pair (i, j) on k-th path
        self.calculate_flatten_mask()
        self.spread = spread

        # Ensure demand has shape (N, N)
        if demand.shape != (N, N):
            demand = demand.reshape((N, N))

        bandwidth_matrix = self.topology.capacity_matrix
        traffic_matrix = demand

        # Compute initial bandwidth utilization
        utilization,utilization_1 = self.calculate_bandwidth_utilization_path(traffic_matrix)
        utilization[bandwidth_matrix == 0] = 0
        # Map (i, j) to row indices in SDtoPath via dictionary
        SD_pair_to_row = self.SD_pair_to_row
        start_time=time.time()
        opt = np.max(utilization_1)
        change_tag=1
        max_utilization_history = []
        time_history = []
        for count in range(epoch):
            
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            max_indices = np.argwhere(rounded_utilization == max_value)
            # pairs = self.get_start_end_array(max_indices)
            pairs = self.get_communication_pairs_for_high_bandwidth_edges(max_indices)
            # if change_tag==1:
            #     pairs=pairs
            #     # print(opt)
            # else:
            #     pairs=pairs[0:min(len(pairs),100)]
                # print(opt)
            def sort_key(pair):
                i, j = pair
                return demand[i, j]
            # pairs = sorted(pairs, key=sort_key, reverse=True)
            # print(1)
            for i, j in pairs:
                time_1=time.time()
                # Use SD_pair_to_row to get row index for (i, j)
                row_idx = SD_pair_to_row[(i, j)]  # Row index from SD_pair_to_row
                # SDtoPath_row = self.SDtoPath[row_idx, :]  # Row corresponding to (i, j)
                
                now_path_list=self.SDtoPath_dict[row_idx]
                if len(now_path_list)==1 :
                    continue

                f_slice = self.f[self.mask_3d]

                flow_on_path_now=f_slice[now_path_list]*traffic_matrix[i, j]

                # Step 4: Map path flow to edge flow (via PathtoEdge sparse mapping)
                for index,iss in enumerate(now_path_list):
                    edge_list=self.PathtoEdge_dict[iss]
                    # print(utilization_1[edge_list]-flow_on_path_now[index]*self.C_r[edge_list])
                    utilization_1[edge_list]=utilization_1[edge_list]-flow_on_path_now[index]*self.C_r[edge_list]           
                
                time_2=time.time()
                # Find maximum utilization
                

                utilization[self.uti_restore] = utilization_1
                u_max_prime0 = np.max(utilization)
                # Update split ratios f_ij
                if demand[i, j] == 0:
                    f_ij0 = np.zeros(len(self.candidate_path[(i, j)]))
                else:
                    f_ij0 = self.update_f_path(utilization, u_max_prime0, i, j, demand)
                
                
                    # Optionally: use Gurobi to optimize split ratios and MLU
                # optimal_u = self.binary_search_optimal_u(utilization, self.update_f_path, i, j, demand,lower_bound=np.max(utilization),upper_bound=opt, target_sum=1, tol=tol)
                # f_ij1 = self.update_f_path(utilization, optimal_u, i, j, demand)
                # f_ij1= f_ij1/np.sum(f_ij1)
                f_ij, u_max_prime = self.update_f_with_gurobi(i, j, traffic_matrix, utilization)
                # print(u_max_prime)
                if demand[i,j]!=0:
                    f_ij = self.update_f_path(utilization, u_max_prime, i, j, demand)
                    f_ij= f_ij/np.sum(f_ij)
                # if demand[i,j]!=0:
                #     import pdb; pdb.set_trace()
                self.f[i, j, :len(f_ij)] = f_ij
                f_slice = self.f[self.mask_3d]
                for index,iis in enumerate(now_path_list) :
                        edge_list=self.PathtoEdge_dict[iis]
                        # print(flow_on_path_now[index],self.C[edge_list])

                        flow_on_path_now=f_slice[now_path_list]*traffic_matrix[i, j]
                        utilization_1[edge_list]=utilization_1[edge_list]+flow_on_path_now[index]*self.C_r[edge_list]
                utilization[self.uti_restore] = utilization_1


                time_3=time.time()
                # print(time_3-time_2,time_2-time_1)
                # if len(now_path_list)==1:
                #     if np.all(utilization_tag == utilization) :
                #         pass
                #     else:
                #         print("utilization changed")

            # Check convergence of optimization
            if opt - np.max(utilization) <= tol * 10:
                break
                if change_tag==0:
                    change_tag = 1
                else:
                    break
            else:
                # import pdb;pdb.set_trace()
                pass
            
            # Update optimal utilization
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)


        # Final recomputation of utilization
        utilization,utilization_1 = self.calculate_bandwidth_utilization_path(traffic_matrix)

        max_value = np.max(utilization)



        return max_value, self.f,max_utilization_history,time_history
    def update_f_with_gurobi(self, i, j, traffic_matrix, utilization_1):
        """
        Optimize path split ratios f[i, j, k] via Gurobi LP while minimizing global MLU.

        Args:
            i, j: Current SD pair.
            traffic_matrix: Current traffic demand matrix.
            utilization_1: Current edge utilization mapping.

        Returns:
            f_ij: Updated split ratio vector.
            u_max_prime: Optimized global MLU value.
        """
        # Get candidate paths
        paths = self.candidate_path[(i, j)]
        if len(paths) <= 1:
            return np.zeros(len(paths)), np.max(utilization_1)  # If only one path, return zeros and current max utilization

        # Initialize model
        m = grb.Model("update_f_path")
        m.Params.OutputFlag = 0  # Disable Gurobi log output

        # Define variables
        num_paths = len(paths)
        f_vars = m.addVars(num_paths, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name="f")
        u_max_prime = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name="u_max_prime")

        # Constraint 1: Sum of path split ratios equals 1
        m.addConstr(grb.quicksum(f_vars[k] for k in range(num_paths)) == 1, "sum_f")

        # Constraint 2: Update utilization on edges for current SD pair's paths
        for k, path in enumerate(paths):
            edges = [(path[idx], path[idx + 1]) for idx in range(len(path) - 1)]
            for edge in edges:
                # Get current edge utilization and capacity
                edge_util = utilization_1[edge]
                capacity = self.topology.capacity_matrix[edge[0], edge[1]]

                # Add constraint: updated edge utilization
                m.addConstr(
                    f_vars[k] * traffic_matrix[i, j]/capacity + edge_util <= u_max_prime,
                    f"utilization_{k}_{edge}"
                )

        # Constraint 3: Apply global bound to all other edges
        all_edges = self.topology.edges  # List of all edges
        for edge in all_edges:
            if edge not in utilization_1:
                continue  # Skip edges not present in utilization mapping
            edge_util = utilization_1[edge]
            capacity = self.topology.capacity_matrix[edge[0], edge[1]]
            # Global constraint: edge utilization must be <= u_max_prime
            m.addConstr(
                edge_util <= u_max_prime,
                f"global_utilization_{edge}"
            )

        # Objective: minimize global MLU
        m.setObjective(u_max_prime, grb.GRB.MINIMIZE)

        # Optimize
        m.optimize()

        # Extract optimal solution
        if m.status == grb.GRB.Status.OPTIMAL:
            f_ij = np.array([f_vars[k].X for k in range(num_paths)])
            u_max_prime_value = u_max_prime.X
        else:
            f_ij = np.zeros(num_paths)  # No solution: return zeros
            u_max_prime_value = np.max(utilization_1)  # Return current max utilization

        return f_ij, u_max_prime_value
    def multi_update_f(self, utilization_list, u_max_prime, i, j, demand_list):
        """
        Find utilization matrix that minimizes each component of f_ij.

        Args:
            utilization_list: List of edge utilization matrices.
            u_max_prime: Maximum utilization threshold.
            i, j: Nodes i and j for the SD pair to update.
            demand_list: List of traffic demand matrices.

        Returns:
            optimal_f_ij: Element-wise minimum across f_ij vectors.
        """
        optimal_f_ij = None

        # Initialize to +inf for comparison with each f_ij component
        min_f_ij = None

        # Iterate over each utilization matrix
        for utilization, demand in zip(utilization_list, demand_list):
            # Compute f_ij
            f_ij = self.update_f_path(utilization, u_max_prime, i, j, demand)

            # Initialize min_f_ij in the first iteration
            if min_f_ij is None:
                min_f_ij = np.full(f_ij.shape, float('inf'))

            # Update min_f_ij by taking element-wise minima
            min_f_ij = np.minimum(min_f_ij, f_ij)

        return min_f_ij


    def multi_rapid_programming_path(self, demand_list, spread, epoch, initial_f=None, tol=1e-7):
        N = self.topology.number_of_nodes()
        change_tag = 0
        # Initialize path split-ratio matrix f
        if initial_f is None:
            self.f = self.initialize_flow_split_matrix_randomly()
        else:
            self.f = initial_f
        # f[i, j, k] denotes fraction of pair (i, j) on k-th path
        self.calculate_flatten_mask()
        self.spread = spread

        # Ensure each demand has shape (N, N)
        for i, demand in enumerate(demand_list):
            if demand.shape != (N, N):
                demand_list[i] = demand.reshape((N, N))

        bandwidth_matrix = self.topology.capacity_matrix
        traffic_matrix_list = demand_list

        # Compute initial bandwidth utilization
        utilization_list = [self.calculate_bandwidth_utilization_path(traffic_matrix) for traffic_matrix in
                            traffic_matrix_list]

        # Map (i, j) to SDtoPath row indices
        SD_pair_to_row = self.SD_pair_to_row

        opt = max(np.max(utilization) for utilization in utilization_list)
        min_opt = opt
        min_f = self.f
        for count in range(epoch):
            max_value = max([np.max(np.round(utilization, 6)) for utilization in utilization_list])
            max_indices = []
            for utilization in utilization_list:
                max_indices_per = np.argwhere(np.round(utilization, 6) == max_value)

                # Collect max_indices for all matrices
                max_indices.append(max_indices_per)
            max_indices = np.vstack(max_indices)
            # pairs = self.get_start_end_array(max_indices)
            pairs = self.get_communication_pairs_for_high_bandwidth_edges(max_indices)

            for i, j in pairs:
                # Use SD_pair_to_row to get row index for (i, j)
                row_idx = SD_pair_to_row[(i, j)]  # Row index from SD_pair_to_row
                SDtoPath_row = self.SDtoPath[row_idx, :]  # Row for (i, j)
                relevant_paths = np.where(SDtoPath_row == 1)[0]  # Paths used by this pair
                for index, traffic_matrix in enumerate(traffic_matrix_list):
                    # Compute flow on relevant paths for this pair
                    FlowOnPath = (self.SDtoPath[row_idx, :].T * traffic_matrix[i, j]) * self.f[self.mask_3d]
                    # Map path flow to edges via PathtoEdge
                    FlowOnEdge = self.PathtoEdge.T @ FlowOnPath

                    # Remove this pair's contribution from utilization
                    utilization_list[index] -= restore_flattened_to_original(FlowOnEdge / self.C, traffic_matrix.shape,
                                                                            self.mask)
                    # utilization -= FlowOnEdge / self.C

                # Find maximum utilization
                u_max_prime = max([np.max(utilization) for utilization in utilization_list])

                # Update split ratios f_ij
                f_ij = self.multi_update_f(utilization_list, u_max_prime, i, j, traffic_matrix_list)

                # Ensure split ratios sum to 1
                if f_ij.sum() >= 1:
                    optimal_u = self.binary_search_optimal_u(utilization_list, self.multi_update_f, i, j,
                                                            traffic_matrix_list, tol=tol)
                    f_ij = self.multi_update_f(utilization_list, optimal_u, i, j, traffic_matrix_list)
                    f_ij = f_ij / f_ij.sum()  # Normalize
                    self.f[i, j, :len(f_ij)] = f_ij
                    for utilization, traffic_matrix in zip(utilization_list, traffic_matrix_list):
                        # Recompute this pair's path flow contribution
                        FlowOnPath = (self.SDtoPath[row_idx, :].T * traffic_matrix[i, j]) * self.f[self.mask_3d]

                        # Re-update edge utilization
                        FlowOnEdge = self.PathtoEdge.T @ FlowOnPath
                        utilization += restore_flattened_to_original(FlowOnEdge / self.C, traffic_matrix.shape,
                                                                    self.mask)

                else:
                    # Binary search optimal split ratio and adjust
                    optimal_u = self.binary_search_optimal_u(utilization_list, self.multi_update_f, i, j,
                                                            traffic_matrix_list, upper_bound=opt,
                                                            lower_bound=u_max_prime, target_sum=1, tol=tol)

                    f_ij = self.multi_update_f(utilization_list, optimal_u, i, j, traffic_matrix_list)
                    f_ij = f_ij / f_ij.sum()  # Normalize
                    self.f[i, j, :len(f_ij)] = f_ij
                    for utilization, traffic_matrix in zip(utilization_list, traffic_matrix_list):
                        # Recompute this pair's path flow contribution
                        FlowOnPath = (self.SDtoPath[row_idx, :].T * traffic_matrix[i, j]) * self.f[self.mask_3d]

                        # Re-update edge utilization
                        FlowOnEdge = self.PathtoEdge.T @ FlowOnPath
                        utilization += restore_flattened_to_original(FlowOnEdge / self.C, traffic_matrix.shape,
                                                                    self.mask)

            # Check convergence
            if abs(opt - max(np.max(utilization) for utilization in utilization_list)) <= tol * 10 or opt - max(
                    np.max(utilization) for utilization in utilization_list) <= 0:
                if opt <= min_opt:
                    min_opt = opt
                    min_f = self.f
                change_tag += 1
                max_indices = []
                if change_tag >= 10:
                    break
                for utilization in utilization_list:
                    max_indices_per = np.argwhere(np.round(utilization, 6) == max_value)

                    # Append max_indices to combined list
                    max_indices.append(max_indices_per)
                max_indices = np.vstack(max_indices)
                # Use the first column as the start node
                start_points = max_indices[:, 0]

                # Find the most frequent start node via numpy unique/bincount
                unique, counts = np.unique(start_points, return_counts=True)
                most_frequent_start = unique[np.argmax(counts)]
                # Get rows whose start matches most_frequent_start
                matching_rows = max_indices[max_indices[:, 0] == most_frequent_start]

                import random
                # Ensure enough nonzero entries to sample
                if len(matching_rows) >= 2:
                    selected_values = random.sample(list(matching_rows), 2)
                else:
                    selected_values = [matching_rows, matching_rows]
                s, d1 = selected_values[0]
                s, d2 = selected_values[1]
                index1 = index2 = 0
                # d1 = 0
                # d2 = 1
                for path_index, path in enumerate(self.candidate_path[(s, d1)]):
                    if d2 in path:
                        index1 = path_index
                for path_index, path in enumerate(self.candidate_path[(s, d2)]):
                    if d1 in path:
                        index2 = path_index

                tmp = self.f[s, d1, 0]
                self.f[s, d1, 0] = self.f[s, d1, index1]
                self.f[s, d1, index1] = tmp
                tmp = self.f[s, d2, 0]
                self.f[s, d2, 0] = self.f[s, d2, index1]
                self.f[s, d2, index2] = tmp

            # Update best utilization
            utilization_list = [self.calculate_bandwidth_utilization_path(traffic_matrix) for traffic_matrix in
                                traffic_matrix_list]
            opt = max(np.max(utilization) for utilization in utilization_list)
        self.f = min_f
        # Recompute utilization one last time
        utilization_list = [self.calculate_bandwidth_utilization_path(traffic_matrix) for traffic_matrix in
                            traffic_matrix_list]
        max_value = max(np.max(utilization) for utilization in utilization_list)

        return max_value, self.f

    def worker_multi_rapid_programming(self, demand_list, spread, epoch, tol):
        """
        Each process independently calls multi_rapid_programming_path to perform computation.

        Args:
            self: Reference to class instance.
            demand_list: List of demand matrices.
            spread: Spread factor.
            epoch: Number of iterations.
            tol: Tolerance value.

        Returns:
            max_value: Best value found.
            f: Corresponding optimal split-ratio matrix.
        """
        # Directly call multi_rapid_programming_path (internally generates f randomly)
        max_value, optimal_f = self.multi_rapid_programming_path(demand_list, spread, epoch, tol=tol)
        return max_value, optimal_f

    def multi_process_multi_rapid_programming(self, demand_list, spread, epoch, num_processes=30, tol=1e-7):
        """
        Use multiprocessing to repeatedly call multi_rapid_programming_path.

        Args:
            demand_list: List of demand matrices.
            spread: Spread factor.
            epoch: Number of iterations.
            num_processes: Number of parallel processes.
            tol: Tolerance value.

        Returns:
            best_max_value: Best max_value found across all processes.
            best_f: Corresponding optimal split-ratio matrix.
        """
        import multiprocessing as mp
        # Initialize process pool
        pool = mp.Pool(processes=num_processes)

        # Start parallel computation; each process independently calls multi_rapid_programming_path
        results = [pool.apply_async(self.worker_multi_rapid_programming, args=(demand_list, spread, epoch, tol))
                   for _ in range(num_processes)]

        # Close pool and wait for all processes to finish
        pool.close()
        pool.join()

        # Gather results from all processes
        results = [res.get() for res in results]

        # Find the best max_value and corresponding split-ratio matrix
        best_max_value, best_f = min(results, key=lambda x: x[0])

        return best_max_value, best_f

    def rapid_programming(self, demand, spread, epoch, tol=1e-7, track_opt_changes=False):
        N = self.topology.number_of_nodes()
        f = self.initialize_f()
        min_f = f
        self.spread = spread

        # Ensure demand matrix is reshaped correctly
        demand = demand.reshape((N, N)) if demand.shape != (N, N) else demand

        bandwidth_matrix = self.topology.capacity_matrix
        traffic_matrix = demand
        utilization = self.calculate_bandwidth_utilization(f, traffic_matrix, bandwidth_matrix)

        opt = np.max(utilization)
        change_tag, min_opt = 0, opt


        # If tracking opt changes, initialize a list to store opt values and iteration times
        opt_changes = [] if track_opt_changes else None
        iteration_times = [] if track_opt_changes else None

        # Record start time
        start_time = time.time()

        for count in range(epoch):
            max_value = np.max(np.round(utilization, 8))
            pairs = self.get_start_end_array(np.argwhere(np.round(utilization, 8) == max_value))

            for i, j in pairs:

                no_i, no_j = np.arange(N) != i, np.arange(N) != j
                # Reduce utilization for pair i, j
                utilization[i, no_i] -= traffic_matrix[i, j] * f[i, j, no_i] / (bandwidth_matrix[i, no_i]+ 1e-5)
                utilization[no_j, j] -= traffic_matrix[i, j] * f[i, j, no_j] / (bandwidth_matrix[no_j, j]+ 1e-5)
                # utilization[bandwidth_matrix == 0] = 0
                u_max_prime = max(np.max(utilization[i, no_i]), np.max(utilization[no_j, j]))
                f_ij = self.update_f(utilization, u_max_prime, i, j, demand)
                if f_ij.sum() >= 1:
                    optimal_u = self.binary_search_optimal_u(utilization, self.update_f, i, j, demand, tol=tol*10)
                    f_ij = self.update_f(utilization, optimal_u, i, j, demand) / f_ij.sum()
                else:
                    optimal_u = self.binary_search_optimal_u(utilization, self.update_f, i, j, demand, upper_bound=opt,
                                                             lower_bound=u_max_prime, target_sum=1, tol=tol*10)
                    f_ij = self.update_f(utilization, optimal_u, i, j, demand) / (f_ij.sum() or 1)

                f[i, j, :] = f_ij / (f_ij.sum() or 1) if f_ij.sum() != 0 else np.zeros_like(f_ij)
                if np.min(f) < 0:
                    print(1)
                    import pdb;pdb.set_trace()
                # Revert utilization changes for updated f_ij
                utilization[i, no_i] += traffic_matrix[i, j] * f[i, j, no_i] / (bandwidth_matrix[i, no_i] + 1e-5)
                utilization[no_j, j] += traffic_matrix[i, j] * f[i, j, no_j] / (bandwidth_matrix[no_j, j] + 1e-5)
            # print(f"opt:{opt},mlu:{np.max(utilization)}")
            opt0 = np.max(utilization)
            if track_opt_changes:
                elapsed_time = time.time() - start_time  # Time delta after each iteration
                opt_changes.append((count, opt0))
                iteration_times.append((count, elapsed_time))  # Track iteration count and runtime since start
            if opt - np.max(utilization) <= tol:
                if np.max(utilization) <= min_opt:
                    min_opt = np.max(utilization)
                    min_f = copy.deepcopy(f)
                change_tag += 1
                if change_tag >= 0:
                    break

                # Select max utilization pairs for swap optimization
                max_indices = np.argwhere(np.round(utilization, 6) == np.max(np.round(utilization, 6)))
                most_frequent_start = np.bincount(max_indices[:, 0]).argmax()
                matching_rows = max_indices[max_indices[:, 0] == most_frequent_start]

                selected_values = random.sample(list(matching_rows), min(2, len(matching_rows)))
                if len(selected_values) == 2:
                    i, j, k = selected_values[0][0], selected_values[0][1], selected_values[1][1]
                    no_i = np.arange(N) != i
                    no_j = np.arange(N) != i
                    no_k = np.arange(N) != k

                    # Adjust utilization for pairs (i,j) and (i,k)
                    for x, j0, no_j0 in [(i, j, no_j), (i, k, no_k)]:
                        utilization[i, no_i] -= traffic_matrix[i, j0] * f[i, j0, no_i] / (
                                    bandwidth_matrix[i, no_i] + 1e-10)
                        utilization[no_j0, j0] -= traffic_matrix[i, j0] * f[i, j0, no_j0] / (
                                    bandwidth_matrix[no_j0, j0] + 1e-10)


                    # Swap f values
                    temp1, temp2 = f[i, j, j], f[i, k, k]
                    f[i, j, j], f[i, k, k] = f[i, j, k], f[i, k, j]
                    f[i, j, k], f[i, k, j] = temp1, temp2

                    for x, j0, no_j0 in [(i, j, no_j), (i, k, no_k)]:
                        utilization[i, no_i] += traffic_matrix[i, j0] * f[i, j0, no_i] / (
                                bandwidth_matrix[i, no_i] + 1e-10)
                        utilization[no_j0, j0] += traffic_matrix[i, j0] * f[i, j0, no_j0] / (
                                bandwidth_matrix[no_j0, j0] + 1e-10)

            opt = np.max(utilization)



        # Final utilization
        utilization = self.calculate_bandwidth_utilization(min_f, demand, bandwidth_matrix)



        # Record end time
        end_time = time.time()

        # Compute total runtime
        total_runtime = end_time - start_time

     
        return np.max(utilization), min_f

    def generate_path_weight_routing(self, f):
        N = self.topology.number_of_nodes()
        path_weight_routing = {}

        for i in range(N):
            for j in range(N):
                if i != j:
                    for k in range(N):
                        if f[i, j, k] >= 0 and i != k:
                            if k == j:
                                # Direct path
                                path_name = f"w_{i}_{j}_{0}"
                            else:
                                # Bypass path
                                path_name=f"{i}{j}{k}"
                                tag=True
                                for index,path in enumerate(self.candidate_path[(i, j)]):
                                    if [i,k,j] == path:
                                        path_name = f"w_{i}_{j}_{index}"
                                        tag=False
                                        break
                                if tag ==True and f[i, j, k]!=0:
                                    print(1)
                                    import pdb;pdb.set_trace
                                # path_name = f"w_{i}_{j}_{k}"
                            path_weight_routing[path_name] = f[i, j, k]

        return path_weight_routing

    def Spread_traffic_engineering(self, demand, Spread):
        """Compute the traffic engineering solutions for a single demand to minimize the worst-case MLU with Google's hedging mechanism.

        Args:
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
            Spread: the spread factor
        """
        if self.candidate_path == None:
            self.renew()
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        else:
            demand = demand
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0

        mlu = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i in range(self.topology.number_of_nodes())
                            for j in range(self.topology.number_of_nodes())
                            if j != i
                            for k in range(len(self.candidate_path[(i, j)]))
                            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        for i in range(self.topology.number_of_nodes()):
            for j in range(self.topology.number_of_nodes()):
                if demand[i, j] == 0 and i != j:
                    m.addConstr(
                        (path_weight[f'w_{i}_{j}_{0}'] == 1 ))
                       
                    m.addConstrs(
                        (path_weight[f'w_{i}_{j}_{k}'] == 0 for k in range(1,len(self.candidate_path[(i, j)]))),
                        name=f"constr_{i}_{j}"
                    )
        
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i 
        )
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
            ) <= mlu * self.topology.edges[edge]['capacity']
            for edge in self.topology.edges
        )

        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.Params.TimeLimit = 2000
        m.optimize()

        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            path_weight_routing = {}
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if i!=j:
                        path_weight_routing[f'w_{i}_{j}_{0}'] = 1
                        for k in range(1,len(self.candidate_path[(i, j)])):
                            path_weight_routing[f'w_{i}_{j}_{k}'] = 0
                    
            return 0 ,path_weight_routing
    
    
            
    def traffic_engineering_pop(self, demand, Spread,pop_num):
        """Compute the traffic engineering solutions for a single demand to minimize the worst-case MLU with Google's hedging mechanism.

        Args:
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
            Spread: the spread factor
        """
        if self.candidate_path == None:
            self.renew()
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        else:
            demand = demand
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0

        mlu = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i in range(self.topology.number_of_nodes())
                            for j in range(self.topology.number_of_nodes())
                            if j != i
                            for k in range(len(self.candidate_path[(i, j)]))
                            ]
        
        path_weight = m.addVars(name_path_weight, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='path_weight')
        for i in range(self.topology.number_of_nodes()):
            for j in range(self.topology.number_of_nodes()):
                if i != j:
                    if demand[i, j] == 0 and i != j:
                        m.addConstrs(
                            (path_weight[f'w_{i}_{j}_{k}'] == 0 for k in range(len(self.candidate_path[(i, j)]))),
                            name=f"constr_{i}_{j}"
                        )

        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i and demand[i,j]>0
        )
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
            ) <= mlu * self.topology.edges[edge]['capacity']/pop_num
            for edge in self.topology.edges
        )
        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        m.Params.TimeLimit = 1000 
        print("solve one problem")

        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')


    def Spread_traffic_engineering_demands(self, demands, Spread):
        """Compute the traffic engineering solutions for a single demand to minimize the worst-case MLU with Google's hedging mechanism.

        Args:
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
            Spread: the spread factor
        """

        for i, demand in enumerate(demands):
            if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
                demands[i] = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
            else:
                demands[i] = demand
        m = grb.Model('traffic_engineering_grb')
        # m.Params.OutputFlag = 0

        mlu = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i in range(self.topology.number_of_nodes())
                            for j in range(self.topology.number_of_nodes())
                            if j != i
                            for k in range(len(self.candidate_path[(i, j)]))
                            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )
        for demand in demands:
            m.addConstrs(
                grb.quicksum(
                    path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                ) <= mlu * self.topology.edges[edge]['capacity']
                for edge in self.topology.edges
            )

        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()

        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')

    def dual_oblivious_traffic_engineering(self, demands):
        """Compute the traffic engineering solutions in the oblivious model"""
        if self.candidate_path == None:
            self.renew()
        m = grb.Model('dual_traffic_engineering_grb')
        m.Params.OutputFlag = 0
        ratio = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='obli_ratio')

        edge_dict = {}
        edge_list = []

        for edge in self.topology.edges:
            edge_list.append(edge)

        edge_src_dst_to_k = {}
        for l in range(self.topology.number_of_edges()):
            for (src, dst, k) in self.edge_to_path[edge_list[l]]:
                if (l, src, dst) not in edge_src_dst_to_k:
                    edge_src_dst_to_k[(l, src, dst)] = [k]
                else:
                    edge_src_dst_to_k[(l, src, dst)].append(k)

        for i in range(self.topology.number_of_edges()):
            edge_dict[edge_list[i]] = i
        # Network Configuration
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i in range(self.topology.number_of_nodes())
                            for j in range(self.topology.number_of_nodes())
                            if j != i
                            for k in range(len(self.candidate_path[(i, j)]))
                            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        # split ratio on ratio, more detial in Making Intra-Domain Routing Robust to Changing
        name_f_dict = [f'f_{l}_{i}_{j}'
                       for l in range(self.topology.number_of_edges())
                       for i in range(self.topology.number_of_nodes())
                       for j in range(self.topology.number_of_nodes())
                       if j != i
                       ]
        f_dict = m.addVars(name_f_dict, lb=0, vtype=grb.GRB.CONTINUOUS, name='f_dict')

        # pi in Making Intra-Domain Routing Robust to Changing
        name_pi = [f'pi_{i}_{j}'
                   for i in range(self.topology.number_of_edges())
                   for j in range(self.topology.number_of_edges())]
        pi = m.addVars(name_pi, lb=0, vtype=grb.GRB.CONTINUOUS, name='pi')

        # p in Making Intra-Domain Routing Robust to Changing
        name_p = [f'p_{i}_{j}_{l}'
                  for i in range(self.topology.number_of_nodes())
                  for j in range(self.topology.number_of_nodes())
                  for l in range(self.topology.number_of_edges())]
        p = m.addVars(name_p, lb=0, vtype=grb.GRB.CONTINUOUS, name='p')

        # Network configuration constraints
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )

        # put the path split on edge
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if (l, i, j) in edge_src_dst_to_k.keys():
                        m.addConstr(
                            f_dict[f'f_{l}_{i}_{j}'] == grb.quicksum(
                                path_weight[f'w_{i}_{j}_{k}'] for k in edge_src_dst_to_k[(l, i, j)])
                        )

        # first constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            m.addConstr(
                grb.quicksum(self.topology.edges[edge_list[j]]['capacity'] * pi[f'pi_{l}_{j}'] for j in
                             range(self.topology.number_of_edges())) <= ratio
            )

        # second constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if i != j:
                        if f'f_{l}_{i}_{j}' in f_dict:
                            m.addConstr(
                                f_dict[f'f_{l}_{i}_{j}'] <= p[f'p_{i}_{j}_{l}'] * self.topology.edges[edge_list[l]][
                                    'capacity']
                            )

        # third constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for e in range(self.topology.number_of_edges()):
                    # edge_list[e][0] is the ToTE_src node of edge
                    # edge_list[e][1] is the dst node of edge
                    m.addConstr(
                        pi[f'pi_{l}_{e}'] + p[f'p_{i}_{edge_list[e][0]}_{l}'] - p[f'p_{i}_{edge_list[e][1]}_{l}'] >= 0
                    )

        # fifth constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                m.addConstr(p[f'p_{i}_{i}_{l}'] == 0)

        # fourth and sixth constraint is in addVars lb.

        m.setObjective(ratio, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            return m.objVal, solution
        else:
            print('No solution')

    def dual_cope_traffic_engineering(self, demands, predict_dms, beta):
        if self.candidate_path == None:
            self.renew()
        for idx, demand in enumerate(predict_dms):
            predict_dms[idx] = demand.reshape(self.topology.number_of_nodes(), self.topology.number_of_nodes())
        oblivious_ratio, _ = self.dual_oblivious_traffic_engineering(demands)
        plenalty_ratio = beta * oblivious_ratio
        m = grb.Model('dual_traffic_engineering_grb')
        m.Params.OutputFlag = 0
        ratio = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='obli_ratio')

        edge_dict = {}
        edge_list = []

        for edge in self.topology.edges:
            edge_list.append(edge)

        edge_src_dst_to_k = {}
        for l in range(self.topology.number_of_edges()):
            for (src, dst, k) in self.edge_to_path[edge_list[l]]:
                if (l, src, dst) not in edge_src_dst_to_k:
                    edge_src_dst_to_k[(l, src, dst)] = [k]
                else:
                    edge_src_dst_to_k[(l, src, dst)].append(k)

        for i in range(self.topology.number_of_edges()):
            edge_dict[edge_list[i]] = i
        # Network Configuration
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i in range(self.topology.number_of_nodes())
                            for j in range(self.topology.number_of_nodes())
                            if j != i
                            for k in range(len(self.candidate_path[(i, j)]))
                            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='path_weight')

        # split ratio on ratio, more detial in Making Intra-Domain Routing Robust to Changing
        name_f_dict = [f'f_{l}_{i}_{j}'
                       for l in range(self.topology.number_of_edges())
                       for i in range(self.topology.number_of_nodes())
                       for j in range(self.topology.number_of_nodes())
                       if j != i
                       ]
        f_dict = m.addVars(name_f_dict, lb=0, vtype=grb.GRB.CONTINUOUS, name='f_dict')

        # pi in Making Intra-Domain Routing Robust to Changing
        name_pi = [f'pi_{i}_{j}'
                   for i in range(self.topology.number_of_edges())
                   for j in range(self.topology.number_of_edges())]
        pi = m.addVars(name_pi, lb=0, vtype=grb.GRB.CONTINUOUS, name='pi')

        # p in Making Intra-Domain Routing Robust to Changing
        name_p = [f'p_{i}_{j}_{l}'
                  for i in range(self.topology.number_of_nodes())
                  for j in range(self.topology.number_of_nodes())
                  for l in range(self.topology.number_of_edges())]
        p = m.addVars(name_p, lb=0, vtype=grb.GRB.CONTINUOUS, name='p')

        # Network configuration constraints
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )

        # put the path split on edge
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if (l, i, j) in edge_src_dst_to_k.keys():
                        m.addConstr(
                            f_dict[f'f_{l}_{i}_{j}'] == grb.quicksum(
                                path_weight[f'w_{i}_{j}_{k}'] for k in edge_src_dst_to_k[(l, i, j)])
                        )

        # first constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            m.addConstr(
                grb.quicksum(self.topology.edges[edge_list[j]]['capacity'] * pi[f'pi_{l}_{j}'] for j in
                             range(self.topology.number_of_edges())) <= plenalty_ratio
            )

        # second constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for j in range(self.topology.number_of_nodes()):
                    if i != j:
                        if f'f_{l}_{i}_{j}' in f_dict:
                            m.addConstr(
                                f_dict[f'f_{l}_{i}_{j}'] <= p[f'p_{i}_{j}_{l}'] * self.topology.edges[edge_list[l]][
                                    'capacity']
                            )

        # third constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                for e in range(self.topology.number_of_edges()):
                    # edge_list[e][0] is the ToTE_src node of edge
                    # edge_list[e][1] is the dst node of edge
                    m.addConstr(
                        pi[f'pi_{l}_{e}'] + p[f'p_{i}_{edge_list[e][0]}_{l}'] - p[f'p_{i}_{edge_list[e][1]}_{l}'] >= 0
                    )

        # fifth constraint in Making Intra-Domain Routing Robust to Changing
        for l in range(self.topology.number_of_edges()):
            for i in range(self.topology.number_of_nodes()):
                m.addConstr(p[f'p_{i}_{i}_{l}'] == 0)

        for demand in predict_dms:
            m.addConstrs(
                grb.quicksum(
                    path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
                ) <= ratio * self.topology.edges[edge]['capacity']
                for edge in self.topology.edges
            )

        m.setObjective(ratio, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            return m.objVal, solution
        else:
            print('No solution')

    def Get_path_capacity(self):
        """Get the capacity of the paths."""
        if self.candidate_path == None:
            self.renew()
        path_capacity = {}
        import pdb;pdb.set_trace()

        for src in range(self.topology.number_of_nodes()):
            for dst in range(self.topology.number_of_nodes()):
                if dst != src:
                    for index, path in enumerate(self.candidate_path[(src, dst)]):
                        path_capacity[(src, dst, index)] = min([self.topology.edges[(path[i], path[i + 1])]['capacity'] \
                                                                for i in range(len(path) - 1)])
        return path_capacity

    def Get_MLU(self, path_weights, demand):
        """Get the MLU of the traffic engineering solution.

        Args:
            path_weights: the routing weights of the paths
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
        """
        if self.candidate_path == None:
            self.renew()
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        mlu = 0
        for edge in self.topology.edges:
            edge_lu = 0
            for (src, dst, k) in self.edge_to_path[edge]:
                try:
                    edge_lu += path_weights[f'w_{src}_{dst}_{k}'] * demand[src][dst] / float( self.topology.edges[edge]['capacity'])
                except KeyError as e:
                    print(f"KeyError: {e} not found in path_weights")
                    # Consider alternative handling: skip, use defaults, or terminate.

                edge_lu += path_weights[f'w_{src}_{dst}_{k}'] * demand[src][dst] / float(
                    self.topology.edges[edge]['capacity'])
            mlu = max(mlu, edge_lu)
        return mlu

    def Spread_traffic_engineering_link_faliure(self, demand, Spread, link_faliure_list):
        """Compute the traffic engineering solutions for fault-aware traffic engineering

        Args:
            demand: the traffic demand, shape: (number_of_nodes * number_of_nodes)
            Spread: the spread factor
            link_faliure_list: the list of failed links
        """
        if self.candidate_path == None:
            self.renew()
        if demand.shape != (self.topology.number_of_nodes(), self.topology.number_of_nodes()):
            demand = demand.reshape((self.topology.number_of_nodes(), self.topology.number_of_nodes()))
        else:
            demand = demand
        m = grb.Model('traffic_engineering_grb')
        m.Params.OutputFlag = 0
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i in range(self.topology.number_of_nodes())
                            for j in range(self.topology.number_of_nodes())
                            if j != i
                            for k in range(len(self.candidate_path[(i, j)]))
                            ]
        path_weight = m.addVars(name_path_weight, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='path_weight')
        mlu = m.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='mlu')
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{i}_{j}_{k}'] for k in range(len(self.candidate_path[(i, j)]))
            ) == 1
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i
        )
        m.addConstrs(
            grb.quicksum(
                path_weight[f'w_{src}_{dst}_{k}'] * demand[src][dst] for (src, dst, k) in self.edge_to_path[edge]
            ) <= mlu * self.topology.edges[edge]['capacity']
            for edge in self.topology.edges
        )

        faliure_od_dict = {}
        for edge_id in link_faliure_list:
            # print(list(self.topology.edges)[edge_id])
            for (src, dst, k) in self.edge_to_path[list(self.topology.edges)[edge_id]]:
                # print((ToTE_src,dst,k))
                faliure_od_dict[(src, dst)] = True
                m.addConstr(
                    path_weight[f'w_{src}_{dst}_{k}'] == 0
                )

        m.addConstrs(
            path_weight[f'w_{i}_{j}_{k}'] * (Spread * grb.quicksum(
                self.path_capacity[(i, j, k)] for k in range(len(self.candidate_path[(i, j)]))
            )) <=
            self.path_capacity[(i, j, k)]
            for i in range(self.topology.number_of_nodes())
            for j in range(self.topology.number_of_nodes())
            if j != i and not faliure_od_dict.get((i, j), False)
            for k in range(len(self.candidate_path[(i, j)]))
        )

        m.setObjective(mlu, grb.GRB.MINIMIZE)
        m.optimize()
        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_weight)
            path_weight_routing = {}
            for w_name in name_path_weight:
                path_weight_routing[w_name] = solution[w_name]
            return m.objVal, path_weight_routing
        else:
            print('No solution')

    def initialize_f(self,capacity_matrix=None):
        N = self.topology.number_of_nodes()
        if capacity_matrix is None:
            capacity_matrix = self.topology.capacity_matrix
        else:
            capacity_matrix = capacity_matrix *self.topology.capacity_matrix
        f = np.zeros((N, N, N), dtype=float)

        if self.candidate_path is None:
            # Generate fallback paths when candidate_path is empty
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    elif capacity_matrix[i, j] != 0:
                        f[i, j, j] = 1  # Direct path
                    else:
                        for k in range(N):
                            if capacity_matrix[i, k] != 0 and capacity_matrix[k, j] != 0:
                                f[i, j, k] = 1  # Valid intermediate-node path
                                break
                        else:
                            print(f"No valid path for node pair ({i}, {j})")
        else:
            # When candidate_path is provided, initialize f from it
            for (i, j), paths in self.candidate_path.items():
                if i == j:
                    continue

                direct_path_found = False
                for path in paths:
                    if len(path) == 2 and path[0] == i and path[1] == j:
                        f[i, j, j] = 1  # Direct path
                        direct_path_found = True
                        break

                if not direct_path_found:
                    for path in paths:
                        for k in path[1:-1]:
                            if capacity_matrix[i, k] != 0 and capacity_matrix[k, j] != 0:
                                f[i, j, k] = 1  # Valid intermediate-node path
                                break
                        else:
                            continue
                        break
                    else:
                        print(f"No valid path for node pair ({i}, {j})")
        return f

    def get_start_end_array(self, max_indices):
        # Build all communication pairs to optimize; duplicates allowed
        N = self.topology.number_of_nodes()
        starts = max_indices[:, 0]
        ends = max_indices[:, 1]
        start_counts = Counter(starts)
        end_counts = Counter(ends)
        total_counts = start_counts + end_counts

        # Build and sort all points by total frequency
        sorted_points = sorted(range(N), key=lambda x: total_counts.get(x, 0), reverse=True)

        pairs = []
        points_info = []
        for point, count in start_counts.items():
            points_info.append((count, point, True))  # True indicates start point
        for point, count in end_counts.items():
            if point not in start_counts:  # Add endpoints not present as starts
                points_info.append((count, point, False))  # False indicates endpoint

        # Sort by frequency
        points_info.sort(key=lambda x: (-x[0], x[2]))

        for _, point, is_start in points_info:
            for other_point in sorted_points:
                if other_point != point:  # Avoid pairing with self
                    if is_start:
                        pairs.append((int(point), int(other_point)))  # Start first
                    else:
                        pairs.append((int(other_point), int(point)))  # End second
    
        return pairs

    def update_f(self, utilization, u_max_prime, i, j, demand):
        if demand[i, j] == 0:
            f_ij = np.zeros(self.topology.number_of_nodes())
            return f_ij
        N = self.topology.number_of_nodes()
        # Compute forward/backward split ratios
        f_ij_f = (u_max_prime - utilization[i, :]) * self.topology.capacity_matrix[i, :] / \
                 demand[i, j]
        f_ij_b = (u_max_prime - utilization[:, j]) * self.topology.capacity_matrix[:, j] / \
                 demand[i, j]

        # Ensure f_ij_f[i] is 0 and f_ij_f[j] is +inf
        f_ij_f[i] = 0
        f_ij_f[j] = float("inf")

        # Compute initial f_ij
        f_ij = np.minimum(f_ij_f, f_ij_b)
        f_ij = np.maximum(f_ij, 0)

        # Update f_ij[j]
        f_ij[j] = np.maximum((u_max_prime - utilization[i, j]),0) * self.topology.capacity_matrix[i, j] / \
                  demand[i, j]
        if np.min(f_ij)<0:
            print(1)
            import pdb;pdb.set_trace()
        # Apply mask matrix computed during initialization
        f_ij *= self.mask_matrix[i, j, :]

        return f_ij


    def update_f_path(self, utilization, u_max_prime, i, j, demand):
        """
        Update f[i, j, k] based on per-path edge utilization; each path has multiple edges needing split ratios.

        Args:
            utilization: Current edge-utilization matrix.
            u_max_prime: Maximum utilization threshold.
            i, j: SD pair being updated.
            demand: Traffic demand matrix.

        Returns:
            f_ij: Updated split ratios for paths k from i to j.
        """

        N = self.topology.number_of_nodes()
        # Candidate paths from i to j
        paths = self.candidate_path[(i, j)]
        num_paths = len(paths)  # Number of paths from i to j
        # Initialize f_ij to zeros
        f_ij = np.zeros(num_paths)

        # Iterate each path and adjust ratios based on link capacity/utilization
        for k, path in enumerate(paths):
            # Track minimum split ratio along the path
            min_capacity_util = float('inf')

            # Walk each edge in the path
            for k, path in enumerate(paths):
                node_pairs = [(path[l], path[l + 1]) for l in range(len(path) - 1)]

                # Collect utilizations and capacities for all edges
                link_utilizations = np.array([utilization[node1, node2] for node1, node2 in node_pairs])
                link_capacities = np.array([self.topology.capacity_matrix[node1, node2] for node1, node2 in node_pairs])

                # Compute split ratio per edge on the path
                link_utilizations = u_max_prime - link_utilizations
                path_utilization = link_utilizations * link_capacities / demand[i, j]

                # Ensure split ratios are non-negative
                path_utilization = np.where(demand[i, j] != 0,
                                            link_utilizations * link_capacities / demand[i, j],
                                            0)
                path_utilization = np.maximum(path_utilization, 0)
                # Take minimum split ratio across edges of this path
                f_ij[k] = np.min(path_utilization)

            return f_ij

    def calculate_bandwidth_utilization(self, f, traffic_matrix, bandwidth_matrix):
        N = self.topology.number_of_nodes()
        numerator = np.zeros((N, N))
        for k in range(N):
            middd = f[:, :, k] * traffic_matrix
            numerator[:, k] += np.sum(middd, axis=1)
            numerator[k, :] += np.sum(middd, axis=0)
        # Compute the denominator
        denominator = bandwidth_matrix

        # Avoid division by zero by using np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            utilization = np.where(denominator != 0, numerator / denominator, 0)

        # Handle cases where both numerator and denominator are zero
        utilization = np.where((denominator == 0) & (numerator == 0), 0, utilization)

        return utilization

    def binary_search_optimal_u(self, utilization, update_f, i, j, demand, upper_bound=None, lower_bound=None,
                                target_sum=1,
                                tol=1e-10):
        """
                                Binary search optimal u such that np.sum(update_f(utilization, u)) equals target_sum.

                                Args:
                                    utilization (numpy.ndarray): Utilization array.
                                    update_f (function): Function taking (utilization, u) returning array same shape as utilization.
                                    target_sum (float): Target sum, default 1.
                                    tol (float): Tolerance to stop the search, default 1e-6.

                                Returns:
                                    float: Optimal u value.
        """
        if upper_bound is None:
            upper_bound = np.max(utilization)
            lower_bound = np.min(utilization)

        while upper_bound - lower_bound > tol:
            mid_u = (lower_bound + upper_bound) / 2
            if demand[i, j] == 0:
                lower_bound = upper_bound = 0
                break
            else:
                result = np.sum(update_f(utilization, mid_u, i, j, demand))
            # result = np.sum(update_f(utilization, mid_u, i, j, demand))

            if result > target_sum:
                upper_bound = mid_u
            else:
                lower_bound = mid_u

        return (lower_bound + upper_bound) / 2

    def get_communication_pairs_for_high_bandwidth_edges(self, high_bandwidth_edges):
        """
        Given edges with highest utilization, find all paths containing them and return their SD pairs.

        Args:
            high_bandwidth_edges (list of tuples): Edges with highest utilization, each as (start, end).

        Returns:
            list: Communication pairs whose paths traverse those high-utilization edges.
        """
        # PathtoEdge: |Φ| x |E|; SDtoPath: |Ω| x |Φ|
        PathtoEdge = self.PathtoEdge  # |Φ| x |E|
        SDtoPath = self.SDtoPath  # |Ω| x |Φ|

        # Indices of edges in high_bandwidth_edges
        edge_indices = [self.edge_list[tuple(edge)] for edge in high_bandwidth_edges if tuple(edge) in self.edge_list]

        # Select paths that traverse high-bandwidth edges
        PathtoEdge_csc = PathtoEdge

        # Slice columns to see which paths traverse these edges
        high_bandwidth_paths=[self.EdgeToPath_dict[edge] for edge in edge_indices]
        high_bandwidth_paths = []
        for edge in edge_indices:
            high_bandwidth_paths.extend(self.EdgeToPath_dict[edge] )
        high_bandwidth_pairs = [self.PathtoSD_dict[p] for p in high_bandwidth_paths]

        pairs = [self.row_to_SD_pair[i] for i in high_bandwidth_pairs]
 

        return pairs

    def initialize_flow_split_matrix_with_numpy(self):
        """
        Initialize the split-ratio matrix with numpy.
        If a direct path exists, assign it ratio 1 and others 0;
        otherwise evenly split ratios across available paths.

        Args:
            topology: Topology object with node count and capacity matrix.
            candidate_path (defaultdict): Candidate paths for each node pair.

        Returns:
            f: 3D split-ratio matrix shaped (N, N, max_num_paths), where max_num_paths is max paths per pair.
            flow_ratio_matrix: List of variable names for split ratios such as w_i_j_k.
        """
        topology = self.topology
        candidate_path = self.candidate_path
        N = topology.number_of_nodes()

        # Find max path count across all pairs to size third dimension
        max_num_paths = max(len(paths) for paths in candidate_path.values())

        # Initialize split-ratio matrix
        f = np.zeros((N, N, max_num_paths), dtype=float)

        # Iterate SD pairs and their candidate paths
        for (i, j), paths in candidate_path.items():
            if i == j:
                continue

            num_paths = len(paths)  # Number of paths from i to j
            if num_paths == 0:
                print(f"No candidate paths for pair ({i}, {j})")
                continue

            # Check for direct path
            direct_path_found = False
            for k, path in enumerate(paths):
                if len(path) == 2 and path[0] == i and path[1] == j:
                    # Direct path gets ratio 1, others 0
                    f[i, j, :] = 0  # Reset all path ratios
                    f[i, j, k] = 1  # Direct path ratio
                    direct_path_found = True
                    break  # Stop after finding direct path

            if not direct_path_found:
                # If no direct path, evenly split among paths
                f[i, j, :num_paths] = 1 / num_paths
        return f

    def initialize_flow_split_matrix_randomly(self):
        """
        Randomly initialize split-ratio matrix.
        For each pair (i, j), generate random ratios for candidate paths summing to 1.

        Returns:
            f: 3D split-ratio matrix shaped (N, N, max_num_paths), where max_num_paths is max paths per pair.
        """
        topology = self.topology
        candidate_path = self.candidate_path
        N = topology.number_of_nodes()

        # Find max path count across all pairs to size third dimension
        max_num_paths = max(len(paths) for paths in candidate_path.values())

        # Initialize split-ratio matrix
        f = np.zeros((N, N, max_num_paths), dtype=float)

        # Iterate SD pairs and candidate paths
        for (i, j), paths in candidate_path.items():
            if i == j:
                continue

            num_paths = len(paths)  # Number of paths from i to j
            if num_paths == 0:
                print(f"No candidate paths for pair ({i}, {j})")
                continue

            # Generate num_paths random values and normalize
            random_weights = np.random.rand(num_paths)
            random_weights /= random_weights.sum()  # Ensure ratios sum to 1

            # Assign random ratios into matrix
            f[i, j, :num_paths] = random_weights

        return f

  
    import numpy as np

    def construct_SDtoPath_and_PathtoEdge(self):
        N = self.topology.number_of_nodes()
        all_paths = []
        all_edges = []
        SD_pair_to_row = {}
        self.row_to_SD_pair = {}
        edge_list = {}

        # 1. Collect all paths and edges
        row_index = 0
        for i in range(N):
            for j in range(N):
                # Only add edges that exist in topology
                if i !=j:
                    SD_pair_to_row[(i, j)] = row_index
                    self.row_to_SD_pair[row_index] = (i, j)  # Reverse mapping
                    row_index += 1
                    paths = self.candidate_path[(i, j)]

                    all_paths.extend(paths)

                if self.topology.has_edge(i, j):
                    all_edges.append((i, j))  # Add edge from topology

        edge_list = {edge: idx for idx, edge in enumerate(all_edges)}
        path_to_index = {tuple(path): idx for idx, path in enumerate(all_paths)}

        # 2. Build the SDtoPath matrix (|Ω| x |Φ|)
        num_paths = len(all_paths)
        num_SD_pairs = len(SD_pair_to_row)
        
        # Create zero-initialized sparse matrix
        from scipy.sparse import lil_matrix
        self.SDtoPath = lil_matrix((num_SD_pairs, num_paths), dtype=np.int0)
        
        SDtoPath_dict = {}

    # Batch construct non-zero entries
        self.PathtoSD_dict = {}
        for (i, j), row_idx in SD_pair_to_row.items():
            paths = self.candidate_path[(i, j)]  # Candidate paths
            SDtoPath_dict[row_idx] = []  # Initialize entry
            
            for path in paths:
                path_idx = path_to_index[tuple(path)]  # Column index for path
                self.SDtoPath[row_idx, path_idx] = 1  # Set sparse entry
                SDtoPath_dict[row_idx].append(path_idx)  # Update dict
                self.PathtoSD_dict[path_idx]=row_idx


        # 3. Build the PathtoEdge matrix (|Φ| x |E|)
        num_edges = len(edge_list)

        # Create zero-initialized matrices
        self.EdgeToPath_dict = defaultdict(list)  
        self.PathtoEdge_T_sparse = lil_matrix(( num_edges,num_paths), dtype=np.float32)
        PathtoEdge = np.zeros((num_paths, num_edges), dtype=int)
        PathtoEdge_dict={}
        for path_idx, path in enumerate(all_paths):
            PathtoEdge_dict[path_idx]=[]
            for l in range(len(path) - 1):
                edge = (path[l], path[l + 1])
                edge_idx = edge_list[edge]
                PathtoEdge[path_idx, edge_idx] = 1
                self.PathtoEdge_T_sparse[edge_idx,path_idx ] = 1
                PathtoEdge_dict[path_idx].append(edge_idx)
                self.EdgeToPath_dict[edge_idx].append(path_idx)
                

        return  PathtoEdge, SD_pair_to_row, edge_list,SDtoPath_dict,PathtoEdge_dict

    def calculate_bandwidth_utilization_path(self, traffic_matrix):
        """
        Compute bandwidth utilization using path split ratios f and matrix structures.

        Args:
            traffic_matrix: Traffic demand matrix for each SD pair.

        Returns:
            utilization: Bandwidth utilization matrix per edge.
        """
        # SDtoPath and PathtoEdge already constructed

        # Mask selects non-diagonal (i, j, :) elements
        R = self.f[self.mask_3d]  # Flattened path split ratios
        DM = traffic_matrix[self.mask].flatten()  # Flatten non-diagonal demand

        # Handle zero capacities to avoid division by zero
        C = self.C
        C = np.where(C == 0, 1e-9, C)  # Replace zeros with small value

        # Flow on each path: FlowOnPath = SDtoPath^T × DM ⊙ R


        
        FlowOnPath = np.zeros(len(R))
        # Matrix multiplication
        for i in range(len(DM)):
            
            FlowOnPath[self.SDtoPath_dict[i]] = DM[i]

        # FlowOnPath = self.SDtoPath_sparse.T.dot(DM)

        # Multiply by R matrix
        FlowOnPath = FlowOnPath * R
        
        
       
        FlowOnEdge = self.PathtoEdge_T_sparse @ FlowOnPath
        # Compute bandwidth utilization per edge
        utilization = FlowOnEdge / C

        # Restore original matrix shape
        utilization_full = np.zeros(traffic_matrix.shape)
        utilization_full[self.uti_restore] = utilization

        return utilization_full,utilization

    def calculate_flatten_mask(self):
        f_shape = self.f.shape
        N = f_shape[0]  # f is N x N x K
        K = f_shape[2]  # number of paths

        # Generate 2D mask excluding diagonal
        mask_2d = ~np.eye(N, dtype=bool)

        # Create 3D mask initialized to False
        mask_3d = np.zeros_like(self.f, dtype=bool)

        for i in range(N):
            for j in range(N):
                if mask_2d[i, j]:  # Only non-diagonal entries
                    if (i, j) in self.candidate_path:
                        path_count = len(self.candidate_path[(i, j)])  # Path count
                        if path_count > 0:
                            mask_3d[i, j, :path_count] = True

        # Save masks
        self.mask_3d = mask_3d
        # 2D mask excluding diagonal
        self.mask = ~np.eye(N, dtype=bool)

    def flatten_f(self, f):
        f_shape = f.shape
        N = f_shape[0]  # Assume f is an N x N x K 3D array

        mask_2d = ~np.eye(N, dtype=bool)  # Build 2D mask excluding (i, i)

        # Expand to 3D mask so entries with identical first two dims are filtered in the third dim
        mask_3d = np.repeat(mask_2d[:, :, np.newaxis], f.shape[2], axis=2)

        # Select non-diagonal (i, j, :) entries via the 3D mask
        f_filtered = f[mask_3d]
        # Flatten the non-diagonal portion
        flattened_f = f_filtered.flatten()
        return flattened_f
