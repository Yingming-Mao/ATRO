import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('..','..'))
sys.path.append(os.path.join('..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ToTE_benchmarks.ToTE_helper import parse_args
from ToTE_benchmarks.ToTE_src.Programming_algorithm import ATRO
from ToTE_benchmarks.ToTE_src.ToTE_env import ToTEEnv
from ToTE_benchmarks.ToTE_src.Bypass import calculate_bandwidth_utilization
from ToTE_benchmarks.ToTE_src.No_Bypass import get_R_from_pods
from ToTE_benchmarks.ToTE_src.Structure import init_structure
from ToTE_benchmarks.ToTE_src.Bypass import variable_separation
from ToTE_benchmarks.ToTE_src.basic import plot_max_utilization_over_iterations
from ToTE_benchmarks.ToTE_src.utils import Get_edge_to_path


# Project root
project_root = os.path.abspath(os.path.join(current_dir, '..'))

from src.utils import print_to_txt
from src.config import RESULT_DIR
from tqdm import tqdm
def plot_combined_histograms(data_dict, title, xlabel, ylabel, num_bins=10, round_labels=False):
    # Determine histogram bin range using min/max across datasets
    all_data = np.concatenate(list(data_dict.values()))
    min_value, max_value = all_data.min(), all_data.max()

    # Slightly expand max value to include edge cases
    max_value = max_value * 1.01  # Expand by 1% to include max

    # Define bins
    bins = np.linspace(min_value, max_value, num_bins + 1)
    if round_labels:
        bin_labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(len(bins) - 1)]
    else:
        bin_labels = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}' for i in range(len(bins) - 1)]

    plt.figure(figsize=(14, 10))

    width = (bins[1] - bins[0]) / (len(data_dict) + 1)  # Bar width

    for i, (method, data) in enumerate(data_dict.items()):
        # Compute counts per bin for each dataset
        hist, _ = np.histogram(data, bins=bins)
        plt.bar(bins[:-1] + i * width - width * len(data_dict) / 2, hist, width=width, label=method, alpha=0.7)

    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    # Set X-axis ticks to bin labels
    plt.xticks(bins[:-1] + width / 2, bin_labels, rotation=45, fontsize=20)

    # Set Y-axis range and ticks
    max_hist = max(max(np.histogram(data, bins=bins)[0]) for data in data_dict.values())
    plt.ylim(0, max_hist + 5)
    plt.yticks(fontsize=20)

    # Auto-select legend location
    plt.legend(loc='best', fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout
    plt.show()



def benchmark(props):
    env = ToTEEnv(props)
    candidate_path = env.pij
    edge_to_path = Get_edge_to_path(env.G, candidate_path)

    if props.TE_solver == 'ATRO':
        algorithm = ATRO(props, env.G, candidate_path, edge_to_path)

    # Select dataset and output paths based on mode
    if props.mode == 'test':
        dm_list = env.simulator.test_hist.tms
        opt_list = env.simulator.test_hist.opts
        result_list = []
        result_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'result.txt')
        time_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'time.txt')
   
        # Add save paths for n_matrix and utilization_records
        n_matrix_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'n_matrix.txt')
        utilization_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'utilization_records.txt')
        opt_path = os.path.join(project_root, "Data", props.topo_name, "test", '11.opt')
        # Ensure directory exists
        os.makedirs(os.path.dirname(n_matrix_save_path), exist_ok=True)
    else:
        dm_list = env.simulator.train_hist.tms
        opt_path = os.path.join(project_root, "Data", props.topo_name, "train", '11.opt')
        # Ensure opt output directory exists
        os.makedirs(os.path.dirname(opt_path), exist_ok=True)
    
    opts = []
    atro_times = []
    n_matrices = []  # Store all n_matrix
    all_utilization_records = []  # Store all utilization_records
    
    for index in tqdm(range(len(dm_list))):
        demands = dm_list[index]
        atro_start = time.time()
        # Use tqdm.write instead of print (debug helper)
        
        if props.mode == 'train':
            # Train mode: compute optimal and save
            u_tmp, n_matrix, utilization_records, n_matrix_records, f = algorithm.atro(props.max_ports, demands)
            atro_time = time.time() - atro_start
            opts.append(u_tmp)
        else:
            # Test mode: use existing logic and record metrics
            u_tmp, n_matrix, utilization_records, n_matrix_records, f = algorithm.atro(props.max_ports, demands)
            # import pdb;pdb.set_trace()
            # tqdm.write(f"{index},{u_tmp}")  # Use tqdm.write instead of print
            atro_time = time.time() - atro_start
            opts.append(u_tmp)
            result_list.append(u_tmp/opt_list[index])
            atro_times.append(atro_time)
            # Save n_matrix (flattened)
            n_matrix_flat = n_matrix.flatten()
            n_matrices.append(n_matrix_flat)
            # Save utilization_records (flatten each record)
            all_utilization_records.append([record for record in utilization_records])

    # Save results based on mode
    if props.mode == 'train':
        # Train mode: save optimal values to opt file
        with open(opt_path, 'w') as f:
            for result in opts:
                f.write(f"{result}\n")
    else:
        with open(opt_path, 'w') as f:
            for result in opts:
                f.write(f"{result}\n")
        # Test mode: save ratios and times
        print_to_txt(result_list, result_save_path)
        print_to_txt(atro_times, time_save_path)


if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    benchmark(props)

# Example commands (unchanged)
# python ToTE_benchmarks/ATRO.py --topo_name Facebook_pod_a --TE_solver ATRO --max_ports 16 --base_capacity 3000 --mode test
# python ToTE_benchmarks/ATRO.py --topo_name Topo_16 --TE_solver ATRO --max_ports 32 --base_capacity 5000 --mode test
# python ToTE_benchmarks/ATRO.py --topo_name Random_30 --TE_solver ATRO --max_ports 64 --base_capacity 5000 --mode test
# python ToTE_benchmarks/ATRO.py --topo_name Topo_32 --TE_solver ATRO --max_ports 64 --base_capacity 5000 --mode test
# python ToTE_benchmarks/ATRO.py --topo_name Topo_64 --TE_solver ATRO --max_ports 128 --base_capacity 5000 --mode test
# python ToTE_benchmarks/ATRO.py --topo_name Facebook_rack_a --TE_solver ATRO --max_ports 256 --base_capacity 5000 --mode test
# python ToTE_benchmarks/ATRO.py --topo_name Facebook_rack_b --TE_solver ATRO --max_ports 512 --base_capacity 5000 --mode test
# python ToTE_benchmarks/ATRO.py --topo_name Topo_128 --TE_solver ATRO --max_ports 256 --base_capacity 5000 --mode test
