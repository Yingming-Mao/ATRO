import numpy as np
from collections import Counter
from .utils import dict_to_numpy, create_submatrices, split_until_limit
import time

class RouteTool:
    @classmethod
    def get_traffic(cls, N, f, traffic_matrix):
        return np.array([[sum(f[i, j_prime, j] * traffic_matrix[i, j_prime] for j_prime in range(N)) + \
                          sum(f[i_prime, j, i] * traffic_matrix[i_prime, j] for i_prime in range(N)) if i != j else 0
                          for j in range(N)] for i in range(N)])

    @classmethod
    def calculate_bandwidth_utilization(cls, N, f, traffic_matrix, n_matrix, bandwidth_matrix):
        """
        Compute per-link bandwidth utilization.

        Args:
            N (int): Number of nodes.
            f (np.ndarray): (N, N, N) array of path splits.
            traffic_matrix (dict or np.ndarray): (N, N) demand matrix.
            n_matrix (np.ndarray): (N, N) connectivity matrix (number of links).
            bandwidth_matrix (dict or np.ndarray): (N, N) link bandwidth matrix.

        Returns:
            np.ndarray: (N, N) utilization matrix.

        """
        # Convert n_matrix and bandwidth_matrix to NumPy arrays if they aren't already
        n_matrix = np.asarray(n_matrix)
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)
        numerator = np.zeros((N, N))

        for k in range(N):
            middd = f[:, :, k] * traffic_matrix
            numerator[:, k] += np.sum(middd, axis=1)
            numerator[k, :] += np.sum(middd, axis=0)
        # Compute the denominator
        # import pdb;pdb.set_trace()
        denominator = n_matrix * bandwidth_matrix
        # Avoid division by zero by using np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            utilization = np.where(denominator != 0, numerator / denominator, 0)
        # Handle cases where both numerator and denominator are zero
        utilization = np.where((denominator == 0) & (numerator == 0), 0, utilization)
        return utilization

    @classmethod
    def update_n_matrix_based_on_utilization(cls, N, R, R_c, n_matrix, utilization,T_tmp,d_wave):
        """
        Increase link counts on the most utilized pairs while respecting port budgets.
        Args:
            N (int): Number of nodes.
            R (list or np.ndarray): Remaining port budget per node.
            R_c (list or np.ndarray): Current port usage per node.
            n_matrix (np.ndarray): (N, N) connectivity matrix.
            utilization (np.ndarray): (N, N) utilization matrix.
        Returns:
            None.
        """
        flattened_matrix = utilization.flatten()
        sorted_indices = np.argsort(-flattened_matrix)
        sorted_indices_2d = np.unravel_index(sorted_indices, utilization.shape)
        sorted_index_tuples = list(zip(*sorted_indices_2d))
        
        # Track remaining degree budget while iterating
        R_remain = [i - j for i, j in zip(R, R_c)]
        R_remain_array = np.array(R_remain)
        
        # Limit how many links we expand
        processed_links = 0
        max_links_to_process = 1000000
        
        for index in sorted_index_tuples:
            # Skip self loops
            if index[0] == index[1]:
                continue
                
            # Ensure at least two nodes still have budget
            if np.sum(R_remain_array > 0) <= 1:
                break
                
            # Both endpoints must have spare ports
            if R_remain_array[index[0]] > 0 and R_remain_array[index[1]] > 0:
                # Compute how many parallel links can be added
                line_plus = min(R_remain_array[index[0]], R_remain_array[index[1]])
                
                # Update topology and degree counters
                n_matrix[index] += line_plus
                n_matrix[index[1], index[0]] += line_plus
                R_c[index[0]] += line_plus
                R_c[index[1]] += line_plus
                
                # Update remaining degree budget
                R_remain_array[index[0]] -= line_plus
                R_remain_array[index[1]] -= line_plus
                
                # Count processed links
                processed_links += 1
                
                # Early exit if limit reached
                if processed_links >= max_links_to_process:
                    break

    @classmethod
    def initialize_f(cls, N, n_matrix):
        """
        Initialize a 3D routing tensor f for the given topology.

        Returns:
            f (np.ndarray): (N, N, N) array of path splits.

        """
        f = np.zeros((N, N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i == j:
                    pass
                elif n_matrix[i, j] != 0:
                    for k in range(N):
                        if k == j:
                            f[i, j, k] = 1  # Direct path
                else:
                    found = False
                    for k in range(N):
                        if n_matrix[i, k] != 0 and n_matrix[k, j] != 0 and found == False:
                            f[i, j, k] = 1  # Found a valid intermediate node
                            found = True
        return f

    @classmethod
    def binary_search_optimal_u(cls, utilization, update_f, i, j, traffic, mask_matrix, upper_bound=None,
                                lower_bound=None, target_sum=1,
                                tol=1e-10):
        """
            Binary search for u such that np.sum(update_f(utilization, u)) matches target_sum.

            Args:
            utilization (numpy.ndarray): Utilization array.
            update_f (function): A function that takes (utilization, u) as input and returns an array of the same shape as utilization.
            target_sum (float): Target sum, default is 1.
            tol (float): Tolerance for stopping condition, default is 1e-6.

            Returns:
            float: The optimal value of u.
            """
        if upper_bound is None:
            upper_bound = np.max(utilization)
            lower_bound = np.min(utilization)

        while upper_bound - lower_bound > tol:
            mid_u = (lower_bound + upper_bound) / 2
            if np.min(update_f(utilization, mid_u, i, j, traffic, mask_matrix))<0:
                import pdb;pdb.set_trace()
                
            result = np.sum(update_f(utilization, mid_u, i, j, traffic, mask_matrix))

            if result > target_sum:
                upper_bound = mid_u
            else:
                lower_bound = mid_u

        return (lower_bound + upper_bound) / 2

    @classmethod
    def lp_rapid(cls, n_matrix, bandwidth_matrix, traffic_matrix, N, epoch, f, mask_matrix,time_limit, tol=1e-5):
        """
        Fast iterative solver for routing matrix f; returns f and the maximum utilization.

        Args:
            n_matrix (np.ndarray): (N, N) connectivity matrix.
            bandwidth_matrix (dict or np.ndarray): (N, N) bandwidth matrix.
            traffic_matrix (dict or np.ndarray): (N, N) demand matrix.
            N (int): Node count.
            epoch (int): Number of iterations.
            tol (float, optional): Convergence tolerance. Defaults to 1e-5.

        Returns:
            Tuple[np.ndarray, float]: (routing matrix f, max utilization).

        """

        def update_f(utilization, u_max_prime, i, j, demand, mask_matrix):
            if demand[i, j] == 0:
                f_ij = np.zeros(utilization.shape[0])
                return f_ij
            # Compute forward/backward flow splits
            f_ij_f = (u_max_prime - utilization[i, :]) * bandwidth_matrix[i, :] / \
                     demand[i, j]
            f_ij_b = (u_max_prime - utilization[:, j]) * bandwidth_matrix[:, j] / \
                     demand[i, j]

            # Ensure source is zero and destination unconstrained
            f_ij_f[i] = 0
            f_ij_f[j] = float("inf")

            # Initial split ratios
            f_ij = np.minimum(f_ij_f, f_ij_b)
            f_ij = np.maximum(f_ij, 0)

            # Update direct component
            f_ij[j] = np.maximum((u_max_prime - utilization[i, j]), 0) * bandwidth_matrix[i, j] / \
                      demand[i, j]
            if np.min(f_ij) < 0:
                print(1)
                import pdb;pdb.set_trace()
            # Apply precomputed mask
            f_ij *= mask_matrix[i, j, :]

            return f_ij

        def get_start_end_array(max_indices, N):
            """
            Build candidate (src, dst) pairs to optimize from the max-index array.

            """
            # Build all candidate pairs (with duplicates)
            all_points = set(range(N))
            starts = max_indices[:, 0]
            ends = max_indices[:, 1]
            start_counts = Counter(starts)
            end_counts = Counter(ends)
            total_counts = start_counts + end_counts
            # Combine start/end info and mark whether it is a start
            points_info = []
            for point, count in start_counts.items():
                points_info.append((count, point, True))  # True means start
            for point, count in end_counts.items():
                if point not in [p for _, p, _ in points_info]:  # Avoid duplicates
                    points_info.append((count, point, False))  # False means end
            # Sort by frequency; break ties by start/end flag
            points_info.sort(key=lambda x: (-x[0], x[2]), reverse=True)
            # Build dictionary for all points including zero counts
            sorted_points_dict = {point: total_counts.get(point, 0) for point in all_points}
            # Sort by total frequency
            sorted_points = sorted(sorted_points_dict.items(), key=lambda x: x[1], reverse=True)
            # sorted_points = sorted_points_dict.items()
            sorted_points = [i[0] for i in sorted_points]
            pairs = []
            # Build pairs for each point
            for info in points_info:
                _, point, is_start = info
                # Pair with all other points (no self-pairs)
                for other_point in sorted_points:
                    if other_point != point:
                        if is_start:
                            
                            pairs.append((int(point), int(other_point)))  # start, end
                        else:
                            pairs.append((int(other_point), int(point)))  # end after start
            return pairs
        start_time=time.time()
        # f = cls.initialize_f(N, n_matrix)
        mask = (bandwidth_matrix == 0)
        # import pdb;pdb.set_trace()
        # # Locate zeros in bandwidth_matrix
        # mask = (bandwidth_matrix == 0)
        mask_matrix=np.copy(mask_matrix)
        # Clear any routes that rely on zero-capacity links
        for i in range(mask_matrix.shape[0]):
            # For each i, find all j where bandwidth_matrix[i, j] == 0
            j_zeros = np.where(mask[i])[0]
            
            # Disable those routes in mask_matrix
            mask_matrix[i, :, j_zeros] = 0
        for j in range(mask_matrix.shape[0]):
            # For each j, find all i where bandwidth_matrix[i, j] == 0
            i_zeros = np.where(mask[j])[0]
            # import pdb;pdb.set_trace()
            # Disable those routes in mask_matrix
            mask_matrix[:, i_zeros, j] = 0    
        for i in range(mask_matrix.shape[0]):
            for j in range(mask_matrix.shape[0]):
                if bandwidth_matrix[i,j]>0:
                    mask_matrix[i, j, j] = 1 
        # import pdb;pdb.set_trace()
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)

        utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        utilization[mask] = 0
        opt = np.max(utilization)

        for count in range(epoch):
            # print("Searching one iteration")
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            # print(f"Searching iteration max {max_value}")
           
            pairs = get_start_end_array(np.argwhere(np.round(utilization, 6) == max_value),utilization.shape[0])
            # pairs = get_start_end_array(max_indices, N)
            # import pdb;pdb.set_trace()
            for i, j in pairs:
                
                # print("Searching one pair")
                if traffic_matrix[i, j] <= 1e-5:
                    continue
                no_i = (np.arange(N) != i) & ~mask[:,i ]
                no_j = (np.arange(N) != j) & ~mask[:, j]

                if np.min(bandwidth_matrix[no_j, j]) == 0 or np.min(bandwidth_matrix[no_i, i]) == 0:
                    print(1)
                    import pdb;pdb.set_trace()

                utilization[i, no_i] = utilization[i, no_i] - traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] - traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])
                # utilization[mask] = 0
                u_max_prime = max(np.max(utilization[i, no_i]), np.max(utilization[no_j, j]))
                f_ij = update_f(utilization, u_max_prime, i, j, traffic_matrix, mask_matrix)
                if f_ij.sum() >= 1:
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            tol=tol)
                else:
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            upper_bound=opt,
                                                            lower_bound=u_max_prime, target_sum=1, tol=tol)
                f_ij = update_f(utilization, optimal_u, i, j, traffic_matrix, mask_matrix)
                if f_ij.sum() == 0:
                    import pdb;pdb.set_trace()
                f_ij = f_ij / f_ij.sum()
                f[i, j, :] = f_ij
                utilization[i, no_i] = utilization[i, no_i] + traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] + traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])

                # utilization[mask] = 0

            if opt - np.max(utilization) <= tol * 1e-1:
                break
            if time.time()-start_time >= time_limit:
                break
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)
        # import pdb;pdb.set_trace()
        # utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        # utilization[mask] = 0
        # max_value = np.max(utilization)
        # max_utilization_history.append(np.max(utilization))
        # time_history.append(current_time)
        return max_value, f


    @classmethod
    def static_lp_rapid(cls, n_matrix, bandwidth_matrix, traffic_matrix, N, epoch, f, mask_matrix,time_limit, tol=1e-5):
        """
        Quickly solve the routing matrix f via linear iteration and return the
        routing fractions with the maximum bandwidth utilization.

        Args:
            n_matrix (np.ndarray): (N, N) adjacency indicating path counts.
            bandwidth_matrix (dict or np.ndarray): (N, N) link bandwidths.
            traffic_matrix (dict or np.ndarray): (N, N) traffic demand matrix.
            N (int): Total node count.
            epoch (int): Number of iterations.
            tol (float, optional): Convergence tolerance. Default 1e-5.

        Returns:
            Tuple[np.ndarray, float]: Routing matrix f and maximum utilization.

        Raises:
            ValueError: Raised inside update_f when traffic_matrix[i, j] is zero to avoid division by zero.

        """

        def update_f(utilization, u_max_prime, i, j, demand, mask_matrix):
            if demand[i, j] == 0:
                f_ij = np.zeros(utilization.shape[0])
                return f_ij
            # Compute forward/backward feasible splits
            f_ij_f = (u_max_prime - utilization[i, :]) * bandwidth_matrix[i, :] / \
                     demand[i, j]
            f_ij_b = (u_max_prime - utilization[:, j]) * bandwidth_matrix[:, j] / demand[i, j]

            # Ensure self is zero and destination is effectively unbounded
            f_ij_f[i] = 0
            f_ij_f[j] = float("inf")

            # Initial split guess
            f_ij = np.minimum(f_ij_f, f_ij_b)
            f_ij = np.maximum(f_ij, 0)

            # Update the direct path share
            f_ij[j] = np.maximum((u_max_prime - utilization[i, j]), 0) * bandwidth_matrix[i, j] / \
                      demand[i, j]
            if np.min(f_ij) < 0:
                print(1)
                import pdb;pdb.set_trace()
            # Apply the precomputed mask
            f_ij *= mask_matrix[i, j, :]

            return f_ij

        # def update_f(utilization, u_max_prime, i, j):
        #     """
        #     Update the specified row/column of f.
        #
        #     Args:
        #         utilization (np.ndarray): (N, N) bandwidth utilization matrix.
        #         u_max_prime (float): Utilization upper bound.
        #         i (int): Row index to update.
        #         j (int): Column index to update.
        #
        #     Returns:
        #         np.ndarray: Updated slice of f with shape (N,).
        #
        #     Raises:
        #         ValueError: Raised when traffic_matrix[i, j] is zero to avoid division by zero.
        #
        #     """
        #     if traffic_matrix[i, j] == 0:
        #         raise ValueError("traffic_matrix[i, j] cannot be zero to avoid division by zero.")
        #     f_ij_f = (u_max_prime - utilization[i, :]) * n_matrix[i, :] * bandwidth_matrix[i, :] / traffic_matrix[i, j]
        #     f_ij_b = (u_max_prime - utilization[:, j]) * n_matrix[:, j] * bandwidth_matrix[:, j] / traffic_matrix[i, j]
        #     f_ij_f[i] = 0
        #     f_ij_f[j] = float("inf")
        #     f_ij = np.minimum(f_ij_f, f_ij_b)
        #     f_ij = np.maximum(f_ij, 0)
        #     f_ij[j] = (u_max_prime - utilization[i, j]) * n_matrix[i, j] * bandwidth_matrix[i, j] / traffic_matrix[i, j]
        #     return f_ij

        def get_start_end_array(max_indices, N):
            """
            Build the communication pairs to optimize from the max index array.

            Args:
                max_indices (np.ndarray): 2D array of [start, end] pairs at max utilization.
                N (int): Node count.

            Returns:
                list: Pairs to optimize as tuples (start, end).

            """
            # Construct all candidate pairs (with duplicates) to optimize
            all_points = set(range(N))
            starts = max_indices[:, 0]
            ends = max_indices[:, 1]
            start_counts = Counter(starts)
            end_counts = Counter(ends)
            total_counts = start_counts + end_counts
            # Combine start/end info and flag whether it is a start
            points_info = []
            for point, count in start_counts.items():
                points_info.append((count, point, True))  # True means start
            for point, count in end_counts.items():
                if point not in [p for _, p, _ in points_info]:  # Avoid duplicates
                    points_info.append((count, point, False))  # False means end
            # Sort by frequency; tie-breaker by start/end flag
            points_info.sort(key=lambda x: (-x[0], x[2]), reverse=True)
            # Build dictionary for all points including zero counts
            sorted_points_dict = {point: total_counts.get(point, 0) for point in all_points}
            # Sort by total frequency
            sorted_points = sorted(sorted_points_dict.items(), key=lambda x: x[1], reverse=True)
            # sorted_points = sorted_points_dict.items()
            sorted_points = [i[0] for i in sorted_points]
            pairs = []
            # Build pairs for each point
            for info in points_info:
                _, point, is_start = info  # point, frequency, is_start flag
                # Pair with all other points (no self-pairs)
                for other_point in sorted_points:
                    if other_point != point:
                        if is_start:
                            
                            pairs.append((int(point), int(other_point)))  # start first
                        else:
                            pairs.append((int(other_point), int(point)))  # end second
            return pairs
        start_time=time.time()
        # f = cls.initialize_f(N, n_matrix)
        mask = (bandwidth_matrix == 0)
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)
        utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        utilization[mask] = 0
        opt = np.max(utilization)
        max_utilization_history = []
        time_history = []
        for count in range(epoch):
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            max_indices = np.argwhere(rounded_utilization == max_value)
            # nonzero_coords = np.nonzero(traffic_matrix)
            # pairs = list(zip(nonzero_coords[0], nonzero_coords[1]))
            # pairs = get_start_end_array(np.argwhere(np.round(utilization, 6) == max_value),utilization.shape[0])
            pairs = get_start_end_array(max_indices, N)
            for i, j in pairs:
                if traffic_matrix[i, j] == 0:
                    continue
                no_i = (np.arange(N) != i) & ~mask[i, :]
                no_j = (np.arange(N) != j) & ~mask[:, j]

                if np.min(bandwidth_matrix[no_j, j]) == 0 or np.min(bandwidth_matrix[no_i, i]) == 0:
                    print(1)
                    import pdb;pdb.set_trace()

                utilization[i, no_i] = utilization[i, no_i] - traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] - traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])
                # utilization[mask] = 0
                u_max_prime = np.max(utilization)
                f_ij = update_f(utilization, u_max_prime, i, j, traffic_matrix, mask_matrix)
                if sum(f_ij)<=1:
                    optimal_u =u_max_prime
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            upper_bound=opt,
                                                            lower_bound=np.max(utilization), target_sum=1, tol=tol)
                else:
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            upper_bound=np.max(utilization),
                                                            lower_bound=0, target_sum=1, tol=tol)
                    
                
                f_ij = update_f(utilization, optimal_u, i, j, traffic_matrix, mask_matrix)
                if f_ij.sum()==0:
                    import pdb;pdb.set_trace()
                # import pdb;pdb.set_trace()
                f_ij = f_ij / f_ij.sum()
                f[i, j, :] = f_ij
                utilization[i, no_i] = utilization[i, no_i] + traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] + traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])

                # utilization[mask] = 0

            if opt - np.max(utilization) <= tol * 1e-1:
                break
            if time.time()-start_time >= time_limit:
                break
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)
        # utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        # utilization[mask] = 0
        # max_value = np.max(utilization)
        # max_utilization_history.append(np.max(utilization))
        # time_history.append(current_time)
        return max_value, f,max_utilization_history,time_history
    
    @classmethod
    def lp_lp_rapid(cls, n_matrix, bandwidth_matrix, traffic_matrix, N, epoch, f, mask_matrix,time_limit, tol=1e-5):
        """
        Solve routing matrix f via iterative updates and return f with maximum
        bandwidth utilization.

        Args:
            n_matrix (np.ndarray): (N, N) adjacency indicating path counts.
            bandwidth_matrix (dict or np.ndarray): (N, N) link bandwidths.
            traffic_matrix (dict or np.ndarray): (N, N) traffic demand matrix.
            N (int): Total node count.
            epoch (int): Number of iterations.
            tol (float, optional): Convergence tolerance. Default 1e-5.

        Returns:
            Tuple[np.ndarray, float]: Routing matrix f and maximum utilization.

        Raises:
            ValueError: Raised inside update_f when traffic_matrix[i, j] is zero to avoid division by zero.

        """

        def update_f(utilization, u_max_prime, i, j, demand, mask_matrix):
            if demand[i, j] == 0:
                f_ij = np.zeros(utilization.shape[0])
                return f_ij
            # Compute forward/backward feasible splits
            f_ij_f = (u_max_prime - utilization[i, :]) * bandwidth_matrix[i, :] / demand[i, j]
            f_ij_b = (u_max_prime - utilization[:, j]) * bandwidth_matrix[:, j] / demand[i, j]

            # Ensure self is zero and destination is effectively unbounded
            f_ij_f[i] = 0
            f_ij_f[j] = float("inf")

            # Initial split guess
            f_ij = np.minimum(f_ij_f, f_ij_b)
            # f_ij = np.maximum(f_ij, 0)

            # Update the direct path share
            f_ij[j] = (u_max_prime - utilization[i, j]) * bandwidth_matrix[i, j] / \
                      demand[i, j]
            if np.min(f_ij) < -1e-10:
                print(1)
                import pdb;pdb.set_trace()
            # Apply the precomputed mask
            f_ij *= mask_matrix[i, j, :]

            return f_ij

 
        def get_start_end_array(max_indices, N):
            """
            Build the communication pairs to optimize from the max index array.

            Args:
                max_indices (np.ndarray): 2D array of [start, end] pairs at max utilization.
                N (int): Node count.

            Returns:
                list: Pairs to optimize as tuples (start, end).

            """
            # Construct all candidate pairs (with duplicates) to optimize
            all_points = set(range(N))
            starts = max_indices[:, 0]
            ends = max_indices[:, 1]
            start_counts = Counter(starts)
            end_counts = Counter(ends)
            total_counts = start_counts + end_counts
            # Combine start/end info and flag whether it is a start
            points_info = []
            for point, count in start_counts.items():
                points_info.append((count, point, True))  # True means start
            for point, count in end_counts.items():
                if point not in [p for _, p, _ in points_info]:  # Avoid duplicates
                    points_info.append((count, point, False))  # False means end
            # Sort by frequency; tie-breaker by start/end flag
            points_info.sort(key=lambda x: (-x[0], x[2]), reverse=True)
            # Build dictionary for all points including zero counts
            sorted_points_dict = {point: total_counts.get(point, 0) for point in all_points}
            # Sort by total frequency
            sorted_points = sorted(sorted_points_dict.items(), key=lambda x: x[1], reverse=True)
            # sorted_points = sorted_points_dict.items()
            sorted_points = [i[0] for i in sorted_points]
            pairs = []
            # Build pairs for each point
            for info in points_info:
                _, point, is_start = info  # point, frequency, is_start flag
                # Pair with all other points (no self-pairs)
                for other_point in sorted_points:
                    if other_point != point:
                        if is_start:
                            
                            pairs.append((int(point), int(other_point)))  # start first
                        else:
                            pairs.append((int(other_point), int(point)))  # end second
            return pairs
        start_time=time.time()
        # f = cls.initialize_f(N, n_matrix)
        mask = (bandwidth_matrix == 0)
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)
        utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        utilization[mask] = 0
        opt = np.max(utilization)
        max_utilization_history = []
        time_history = []
        for count in range(epoch):
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            max_indices = np.argwhere(rounded_utilization == max_value)
            pairs = get_start_end_array(np.argwhere(np.round(utilization, 6) == max_value),utilization.shape[0])
            # pairs = get_start_end_array(max_indices, N)
            for i, j in pairs:
                if traffic_matrix[i, j] == 0:
                    continue
                no_i = (np.arange(N) != i) & ~mask[i, :]
                no_j = (np.arange(N) != j) & ~mask[:, j]



                utilization[i, no_i] = utilization[i, no_i] - traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] - traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])
                # utilization[mask] = 0

                # u_max_prime0 = np.max(utilization)
                # f_ij2 = update_f(utilization, u_max_prime0, i, j, traffic_matrix, mask_matrix)
                # optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                #                                             upper_bound=opt,
                #                                             lower_bound=u_max_prime0, target_sum=1, tol=tol)
                # Solve path split ratios via Gurobi
                f_ij0, u_max_prime = cls.update_f_with_lp(i, j, traffic_matrix, bandwidth_matrix, utilization, mask_matrix, N)
                # if  abs(optimal_u-u_max_prime)>=1e-5:
                #     import pdb;pdb.set_trace()
                # f_ij = update_f(utilization, u_max_prime, i, j, traffic_matrix, mask_matrix)
                # print(u_max_prime,optimal_u)
                # import pdb;pdb.set_trace()
                # Update f and link utilization
                # f_ij = f_ij / f_ij.sum()
                # if np.sum(f_ij2)<=1:
                #     import pdb;pdb.set_trace()
                    
                f[i, j, :] = f_ij0
                
    
                
                utilization[i, no_i] = utilization[i, no_i] + traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] + traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])

                if time.time()-start_time >= time_limit:
                    print("time limit1")
                    break 
                    
                


                # utilization[mask] = 0

            if opt - np.max(utilization) <= tol * 1e-1:
                break
            if time.time()-start_time >= time_limit:
                print("time limit2")
                break
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)
        # utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        # utilization[mask] = 0
        # max_value = np.max(utilization)

        # max_utilization_history.append(np.max(utilization))
        # time_history.append(current_time)
        return max_value, f,max_utilization_history,time_history
    @classmethod
    def update_f_with_lp(cls, i, j, traffic_matrix, bandwidth_matrix, utilization, mask_matrix, N):
        """
        Optimize path split ratios f[i, j, k] with linear programming while
        minimizing the global MLU.

        Args:
            i, j: Current source/destination pair.
            traffic_matrix: Traffic demand matrix.
            bandwidth_matrix: Bandwidth matrix.
            utilization: Current link utilization matrix.
            mask_matrix: Mask indicating usable paths.
            N: Number of nodes.

        Returns:
            f_ij: Updated split ratios.
            u_max: Optimized maximum link utilization.
        """
        from gurobipy import Model, GRB, quicksum

        # Initialize Gurobi model
        m = Model("update_f")
        m.Params.OutputFlag = 0  # Suppress solver logs

        # Define variables
        f = m.addVars(N, lb=0, ub=1, name="f")  # split ratios
        u = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="u")  # maximum link utilization

        # Objective: minimize global maximum link utilization
        m.setObjective(u, GRB.MINIMIZE)

        # Constraint 1: split ratios sum to 1
        m.addConstr(quicksum(f[k] for k in range(N)) == 1, name="sum_f")

        # Constraint 2: link utilization bounds
        for k in range(N):
            if mask_matrix[i, j, k] == 0:
                m.addConstr(f[k] == 0, name=f"mask_{i}_{j}_{k}")
            else:
                # Dynamically build edges for the path
                if j==k:
                    edges=[(i,j)]
                elif i==k:
                    m.addConstr(
                        f[k] == 0,
                        name=f"util_{i}_{j}_{k}_{edge}"
                    )
                    edges=[]
                
                else:
                    edges = [(i,k),(k,j)]
                    
                
                for edge in edges:
                    edge_utilization = utilization[edge[0], edge[1]]
                    
                    capacity = bandwidth_matrix[edge[0], edge[1]]
                    if capacity==0:
                        import pdb; pdb.set_trace()
                    # Enforce utilization on this edge does not exceed u
                    m.addConstr(
                        f[k] * traffic_matrix[i, j]/capacity + edge_utilization <= u,
                        name=f"util_{i}_{j}_{k}_{edge}"
                    )

        # Constraint 3: u >= utilization on every existing link
        for x in range(N):
            for y in range(N):
                if bandwidth_matrix[x, y] > 0:
                    m.addConstr(
                        u >= utilization[x, y],  # u must dominate current utilization
                        name=f"u_ge_utilization_{x}_{y}"
                    )

        # Optimize
        m.optimize()

        # Extract results
        if m.Status == GRB.OPTIMAL:
            f_ij = np.array([f[k].X if mask_matrix[i, j, k] == 1 else 0 for k in range(N)])
            u_max = u.X
        else:
            print("No feasible solution found.")
            f_ij = np.zeros(N)
            u_max = np.max(utilization)

        return f_ij, u_max
    @classmethod
    def lp_by_gp(cls, n_matrix, s_matrix, d_wave, N, mask_matrix):
        from gurobipy import Model, GRB, quicksum
        f_d = {}
        # Initialize model
        m = Model("NetworkOptimization")
        # Variables
        f = m.addVars(N, N, N, name="f", lb=0)  # flow allocation variables
        u = m.addVar(name="u")  # objective variable

        # Objective
        m.setObjective(u, GRB.MINIMIZE)

        # Constraints (matching the original formulation)

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i == j or i == k:
                        m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")
                    # if mask_matrix[i,j,k]==0:
                    #     m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")
                        

        # Flow conservation: split ratios sum to 1
        for i in range(N):
            for j in range(N):
                if i != j:
                    m.addConstr(quicksum(f[i, j, k] for k in range(N)) == 1, name=f"flow_sum_{i}_{j}")

        # Cost constraints with reduced redundant computation
        cost_sums = {}
        for i in range(N):
            for j in range(N):
                if i != j:
                    outgoing_cost = quicksum(f[i, jp, j] * d_wave[i, jp] for jp in range(N))
                    incoming_cost = quicksum(f[ip, j, i] * d_wave[ip, j] for ip in range(N))
                    cost_sums[(i, j)] = outgoing_cost + incoming_cost
                    m.addConstr(cost_sums[(i, j)] <= u * n_matrix[i][j] * s_matrix[i, j], name=f"cost_{i}_{j}")

        # Solve the model
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 43200 )
        m.optimize()
        f_d = np.zeros((N, N, N))
        # Extract results
        if m.SolCount > 0:
            for key, var in f.items():
                f_d[key] = var.x
            
 
            return u.x, f_d
        else:
            print("No feasible solution found within the time limit.")
            return 100, 100

    @classmethod
    def lp_by_gp_sub(cls, n_matrix, s_matrix, d_wave, N, mask_matrix):
        from gurobipy import Model, GRB, quicksum
        f_d = {}
        # Initialize model
        m = Model("NetworkOptimization")
        non_zero_indices = np.argwhere(d_wave != 0)
        # Variables
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i, j in non_zero_indices
                            for k in range(N)
                            ]
        f = m.addVars(name_path_weight, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='path_weight')
        # f = m.addVars(N, N, N, name="f", lb=0)  # flow allocation variables
        u = m.addVar(name="u")  # objective variable

        # Objective
        m.setObjective(u, GRB.MINIMIZE)

        # Constraints (matching the original formulation)

        for i, j in non_zero_indices:
            for k in range(N):
                if i == j or i == k:
                    m.addConstr(f[f'w_{i}_{j}_{k}'] == 0, name=f"f_zero_{i}_{j}_{k}")

        # Flow conservation: split ratios sum to 1
        for i, j in non_zero_indices:
            if i != j:
                m.addConstr(quicksum(f[f'w_{i}_{j}_{k}'] for k in range(N)) == 1, name=f"flow_sum_{i}_{j}")

        # Cost constraints with reduced redundant computation
        cost_sums = {}
        for i in range(N):
            for j in range(N):
                if i != j:
                    outgoing_cost = quicksum(f[f'w_{i}_{jp}_{j}'] * d_wave[i, jp] for jp in range(N) if
                                             np.any(np.all(non_zero_indices == [i, jp], axis=1)))
                    incoming_cost = quicksum(f[f'w_{ip}_{j}_{i}'] * d_wave[ip, j] for ip in range(N) if
                                             np.any(np.all(non_zero_indices == [ip, j], axis=1)))
                    cost_sums[(i, j)] = outgoing_cost + incoming_cost
                    m.addConstr(cost_sums[(i, j)] <= u * n_matrix[i][j] * s_matrix[i, j], name=f"cost_{i}_{j}")

        # Solve the model
        m.setParam('OutputFlag', 0)
        m.optimize()

        import re
        pattern = r'w_(\d+)_(\d+)_(\d+)'
        # Extract i, j, k
        extracted_values = [re.match(pattern, name).groups() for name in name_path_weight]

        # Convert to integers
        extracted_values = [(int(i), int(j), int(k)) for i, j, k in extracted_values]
        f_d = np.zeros((N, N, N))
        # Extract results
        if m.status == GRB.OPTIMAL:
            index = 0
            for key, var in f.items():
                f_d[extracted_values[index]] = var.x
                index += 1
            return f_d, u.x
        else:
            print("error")

    # @classmethod
    # def lp_by_pop(cls, n_matrix, s_matrix, d_wave, N, pop_number):
    #     import time
    #     s_matrix = dict_to_numpy(s_matrix, N)
    #     d_wave = dict_to_numpy(d_wave, N)

    #     d_pop_list = create_submatrices(d_wave, pop_number)
    #     d_pop_list = split_until_limit(d_pop_list, 0.5, pop_number)

    #     s_pop_list = []
    #     for i in range(len(d_pop_list)):
    #         s_pop_list.append(s_matrix / pop_number)
    #     # ss=np.sum(d_pop_list,axis=0)
    #     # sss=np.sum(d_wave)
    #     f_list = []
    #     u_list = []
    #     for i in range(len(d_pop_list)):
    #         s_time = time.time()
    #         f, u = cls.lp_by_gp_sub(n_matrix, s_pop_list[i], d_pop_list[i], N)
    #         f_list.append(f)
    #         u_list.append(u)
    #         # print(time.time() - s_time)
    #     f = np.sum(f_list, axis=0)
    #     utilization = cls.calculate_bandwidth_utilization(N, f, d_wave, n_matrix, s_matrix)
    #     # utilization[mask] = 0

    #     return np.max(utilization), f
    @classmethod
    def lp_by_pop(cls, n_matrix, s_matrix, d_wave, N, pop_number):
        def process_submatrix(args):
            cls, n_matrix, s_sub, d_sub, N = args
            return cls.lp_by_gp_sub(n_matrix, s_sub, d_sub, N)
        import time
        from concurrent.futures import ProcessPoolExecutor

        s_matrix = dict_to_numpy(s_matrix, N)
        d_wave = dict_to_numpy(d_wave, N)

        d_pop_list = create_submatrices(d_wave, pop_number)
        d_pop_list = split_until_limit(d_pop_list, 0.5, pop_number)

        # Split s_matrix evenly across populations
        s_pop_list = [s_matrix / pop_number for _ in range(len(d_pop_list))]

        # Prepare inputs
        args_list = [(cls, n_matrix, s_pop_list[i], d_pop_list[i], N) for i in range(len(d_pop_list))]

        # Parallel computation
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_submatrix, args_list))

        # Collect results
        f_list, u_list = zip(*results)
        f = np.sum(f_list, axis=0)
        utilization = cls.calculate_bandwidth_utilization(N, f, d_wave, n_matrix, s_matrix)

        return np.max(utilization), f
    @classmethod
    def lp_top_p(cls, n_matrix, s_matrix, d_wave, N, p, mask_matrix):
        import time
        import numpy as np

        # Convert input matrices to numpy
        s_matrix = dict_to_numpy(s_matrix, N)
        d_wave = dict_to_numpy(d_wave, N)

        # Initialize 3D flow matrices
        f_direct = np.zeros((N, N, N))  # store direct flow
        f_optimized = np.zeros((N, N, N))  # store optimized flow

        # Flatten demand and locate the top p%
        flat_indices = np.arange(d_wave.size)
        flat_demands = d_wave.flatten()

        top_p_count = int(len(flat_demands) * p / 100)
        top_p_indices = np.argsort(-flat_demands)[:top_p_count]  # sorted by demand descending

        # Build optimized demand matrix d_top_p
        d_top_p = np.zeros_like(d_wave)
        for idx in top_p_indices:
            i, j = divmod(idx, N)
            d_top_p[i, j] = d_wave[i, j]

        # Build direct-demand matrix d_direct
        d_direct = d_wave - d_top_p

        # Optimize the selected demand portion
        start_time = time.time()
        # f_partial, u = cls.lp_by_gp_sub(n_matrix, s_matrix, d_top_p, N)  # expected to return 3D matrix f_partial
        f_partial, u = cls.lp_by_gp(n_matrix, s_matrix, d_top_p, N, mask_matrix)
        
        # print(f"Optimization Time: {time.time() - start_time}s")

        # Apply optimized result to f_optimized
        f_optimized[:, :, :] = f_partial

        # Handle remaining demand by sending it directly
        for i in range(N):
            for j in range(N):
                if d_direct[i, j] > 0:
                    f_direct[i, j, j] = 1  # direct flow written to f[i, j, j]

        # Combine optimized flow and direct flow
        f_final = f_optimized + f_direct

        # Compute bandwidth utilization
        utilization = cls.calculate_bandwidth_utilization(N, f_final, d_wave, n_matrix, s_matrix)

        return np.max(utilization), f_final




if __name__ == '__main__':
    # # Initialization example
    #
    # pod_count = 5
    # up_link_port_range = [256, 256]
    # up_link_bandwidth_range = [100, 100]
    # traffic_range = [0, 100]
    # error_tolerance = 1e-12
    #
    # pods, traffic_matrix, bandwidth_matrix = init_structure(pod_count, up_link_port_range, up_link_bandwidth_range,
    #                                                         traffic_range, 1)
    #
    # max_bandwidth_direct, direct_matrix, _ = no_by_pass(pod_count, pods, traffic_matrix, bandwidth_matrix,
    #                                                     error_tolerance)
    #
    # R = [pod.R for pod in pods]
    # N = len(pods)
    # u_no_by_pass, n_matrix, path = no_by_pass(N, pods, traffic_matrix, bandwidth_matrix, 0.0001)
    #
    # continue_flag = True
    # u_tmp = float("inf")
    # e = 1e-7
    # utilization_records = []  # Record utilization each iteration
    # n_matrix_records = []  # Record n_matrix each iteration
    # i = 0
    # R = [pod.R for pod in pods]
    #
    # T_tmp = traffic_matrix
    # f = RouteTool.initialize_f(N, n_matrix)
    #
    # while continue_flag:
    #     u_no_by_pass, n_matrix, path = no_by_pass(N, pods, T_tmp, bandwidth_matrix, e)
    #     utilization = RouteTool.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
    #     # utilization = np.clip(utilization, a_min=u_no_by_pass, a_max=u_no_by_pass)
    #     utilization_records.append(utilization.copy())  # Record utilization for this iteration
    #     R_c = calculate_R_c(n_matrix, pods, N)
    #     RouteTool.update_n_matrix_based_on_utilization(N, R, R_c, n_matrix, utilization)
    #     start_time = time.time()
    #     f, u_now = RouteTool.lp_rapid(n_matrix, bandwidth_matrix, traffic_matrix, N, 200, tol=1e-6)
    #     print(f"time:{time.time() - start_time},result:{u_now}")
    #     start_time = time.time()
    #     f1, u_now1 = RouteTool.lp_by_gp(n_matrix, bandwidth_matrix, traffic_matrix, N)
    #     f1 = dict_to_numpy(f1, N)
    #     print(f"time:{time.time() - start_time},result:{u_now1}")
    #     traffic_matrix = dict_to_numpy(traffic_matrix, N)
    #
    #     utilization = RouteTool.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
    #
    #     utilization_records.append(utilization.copy())  # Record utilization for this iteration
    #     if u_tmp - u_now < 1e-7:
    #         continue_flag = False
    #         print(f"Final iteration {i}, current max utilization {u_tmp}")
    #     else:
    #         T_tmp = RouteTool.get_traffic(N, f, traffic_matrix)
    #         u_tmp = u_now
    #         print(f"Iteration {i}, current max utilization {u_tmp}")
    #         i += 1
    #
    #     # utilization_records.append(utilization.copy())  # Record utilization for this iteration
    #     n_matrix_records.append(n_matrix.copy())  # Record n_matrix for this iteration
    #
    # f, max_value = RouteTool.lp_rapid(n_matrix, bandwidth_matrix, traffic_matrix, N, 50)
    pass
