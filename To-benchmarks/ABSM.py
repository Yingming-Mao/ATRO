import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ToTE_benchmarks.ToTE_helper import parse_args
from ToTE_benchmarks.ToTE_src.ToTE_env import ToTEEnv
from ToTE_benchmarks.ToTE_src.Programming_algorithm import COUDER
from ToTE_benchmarks.ToTE_src.utils import Get_edge_to_path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('..','..'))
# Project root
project_root = os.path.abspath(os.path.join(current_dir, '..'))
from src.utils import print_to_txt
from src.config import RESULT_DIR
from tqdm import tqdm
from ToTE_benchmarks.ToTE_src.Programming_algorithm import ATRO


def benchmark(props):
    env = ToTEEnv(props)
    candidate_path = env.pij
    edge_to_path = Get_edge_to_path(env.G, candidate_path)
    
    if props.TE_solver == 'ABSM':
        algorithm = ATRO(props, env.G, candidate_path, edge_to_path)

    # Select dataset split and output paths by mode
    if props.mode == 'test':
        dm_list = env.simulator.test_hist.tms
            # Optionally load opt1 file
        opt1_path = os.path.join(project_root, "Data", props.topo_name, "test", '11.opt1')
        opt_list = []
        if os.path.exists(opt1_path):
            with open(opt1_path, 'r') as f:
                for line in f:
                    opt_list.append(float(line.strip()))
        result_list = []
        result_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'result.txt')
        time_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'time.txt')
        topo_path_ = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'count.txt')
        opt_path = os.path.join(project_root, "Data", props.topo_name, "test", '11.opt1')
        # n_matrix_save_path can be added if persistence is needed
    else:
        dm_list = env.simulator.train_hist.tms
        opt_path = os.path.join(project_root, "Data", props.topo_name, "train", '11.opt1')
        # Ensure training opt directory exists
        os.makedirs(os.path.dirname(opt_path), exist_ok=True)
        
    opts = []
    couder_times = []
    n_matrices = []  # store per-demand n_matrix count
    
    for index in tqdm(range(len(dm_list))):
        demands = dm_list[index]
        atro_start = time.time()
        
        if props.mode == 'train':
            # Train mode: integer formulation
            u_tmp, f, n_matrix = algorithm.couder(props.max_ports, demands)
            couder_time = time.time() - atro_start
            # print(f"{index},{u_tmp}")
            opts.append(u_tmp)
        else:
            # Test mode: existing logic
            N=np.sqrt(demands.shape[0])
            
            u_tmp,n_matrix_mcf = algorithm.absm(props.max_ports, demands)
            n_matrices.append(np.sum(n_matrix_mcf))
            # import pdb;pdb.set_trace()
            # u_tmp, f,n_matrix = algorithm.couder(props.max_ports, demands)
            couder_time = time.time() - atro_start
            result_list.append(u_tmp/opt_list[index])
            opts.append(u_tmp)
            # print(f"{index},{u_tmp/opt_list[index]}")
            couder_times.append(couder_time)
            
            

    # Save outputs based on mode
    if props.mode == 'train':
        # Train mode: write optimal values
        with open(opt_path, 'w') as f:
            for result in opts:
                f.write(f"{result}\n")
    else:
        # Test mode: save ratios and runtime
        # with open(opt_path, 'w') as f:
        #     for result in opts:
        #         f.write(f"{result}\n")
        print_to_txt(result_list, result_save_path)
        print_to_txt(couder_times, time_save_path)
        print_to_txt(n_matrices, topo_path_)
        
        






if __name__ == '__main__':

    props = parse_args(sys.argv[1:])
    benchmark(props)

# python To-benchmarks/ABSM.py --topo_name Facebook_pod_a --TE_solver ABSM --max_ports 16 --base_capacity 5000 --mode test
# python To-benchmarks/ABSM.py --topo_name Topo_16 --TE_solver ABSM --max_ports 32 --base_capacity 5000 --mode test
# python To-benchmarks/ABSM.py --topo_name Topo_32 --TE_solver ABSM --max_ports 64 --base_capacity 5000 --mode test
# python To-benchmarks/ABSM.py --topo_name Topo_64 --TE_solver ABSM --max_ports 128 --base_capacity 5000 --mode test
# python To-benchmarks/ABSM.py --topo_name Facebook_rack_a --TE_solver ABSM --max_ports 256 --base_capacity 5000 --mode test
# python To-benchmarks/ABSM.py --topo_name Topo_128 --TE_solver ABSM --max_ports 256 --base_capacity 5000 --mode test
# python To-benchmarks/ABSM.py --topo_name Topo_256 --TE_solver ABSM --max_ports 512 --base_capacity 5000 --mode test
# python To-benchmarks/ABSM.py --topo_name Topo_512 --TE_solver ABSM --max_ports 1024 --base_capacity 5000 --mode test