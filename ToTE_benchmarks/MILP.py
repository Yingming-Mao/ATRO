import sys
import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ToTE_benchmarks.ToTE_helper import parse_args
from ToTE_benchmarks.ToTE_src.ToTE_env import ToTEEnv
from ToTE_benchmarks.ToTE_src.Programming_algorithm import COUDER
from ToTE_benchmarks.ToTE_src.utils import Get_edge_to_path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('..','..'))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
from src.utils import print_to_txt
from src.config import RESULT_DIR
from tqdm import tqdm

def benchmark(props):
    env = ToTEEnv(props)
    candidate_path = env.pij
    edge_to_path = Get_edge_to_path(env.G, candidate_path)
    
    if props.TE_solver == 'MILP':
        algorithm = COUDER(props, env.G, candidate_path, edge_to_path)

        # Select dataset split and output locations
    if props.mode == 'test':
        dm_list = env.simulator.test_hist.tms
        opt_list = env.simulator.test_hist.opts
        result_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'result.txt')
        opt_path = os.path.join(project_root, "Data", props.topo_name, "test", '20.opt')
        time_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'time.txt')
        n_matrix_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'n_matrix.txt')
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(n_matrix_save_path), exist_ok=True)
    else:
        dm_list = env.simulator.train_hist.tms
        opt_path = os.path.join(project_root, "Data", props.topo_name, "train", '20.opt')
        time_save_path = os.path.join(RESULT_DIR, props.topo_name, props.TE_solver, 'train_time.txt')

        # Ensure output directories exist
    os.makedirs(os.path.dirname(opt_path), exist_ok=True)
    os.makedirs(os.path.dirname(time_save_path), exist_ok=True)
    
    opts = []
    result_list=[]
    solve_times = []
    n_matrices = []

    for index in tqdm(range(len(dm_list))):
        demands = dm_list[index]
        start_time = time.time()
        # Solve integer formulation
        u_tmp, f, n_matrix = algorithm.couder(props.max_ports, demands, "INTEGER")
        if props.mode == 'test':
            result_list.append(u_tmp / opt_list[index])
        else:
            opts.append(u_tmp)

        solve_time = time.time() - start_time
        solve_times.append(solve_time)

        # Capture flattened n_matrix per demand
        n_matrix_flat = n_matrix.flatten()
        n_matrices.append(n_matrix_flat)
    
    if props.mode == 'test':
        print_to_txt(result_list, result_save_path)
        print_to_txt(solve_times, time_save_path)
        # Save n_matrix data if needed
        # with open(n_matrix_save_path, 'w') as f:
        #     for n_matrix_flat in n_matrices:
        #         line = ' '.join(map(str, n_matrix_flat))
        #         f.write(f"{line}\n")
    else:
        with open(opt_path, 'w') as f:
            for result in opts:
                f.write(f"{result}\n")
        print_to_txt(solve_times, time_save_path)

if __name__ == '__main__':
    props = parse_args(sys.argv[1:])
    # import pdb;pdb.set_trace()
    benchmark(props)

# python ToTE_benchmarks/MILP.py --topo_name Facebook_pod_a --TE_solver MILP --max_ports 16 --base_capacity 3000 --mode test
# python ToTE_benchmarks/MILP.py --topo_name Facebook_pod_b --TE_solver MILP --max_ports 16 --base_capacity 5000 --mode test
# python ToTE_benchmarks/MILP.py --topo_name Topo_16 --TE_solver MILP --max_ports 32 --base_capacity 5000 --mode test
# python ToTE_benchmarks/MILP.py --topo_name Topo_32 --TE_solver MILP --max_ports 64 --base_capacity 5000 --mode test
                # Save flattened n_matrix