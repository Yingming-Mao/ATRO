import argparse

def add_default_args(parser):

    parser.add_argument('--topo_name', type=str, default = 'Facebook_pod_b')
    parser.add_argument('--paths_file', type=str, default = 'tunnels.txt')
    parser.add_argument('--path_num', type=int, default=3)
    
    parser.add_argument('--hist_len', type=int, default = 1)

    parser.add_argument('--TE_solver', type=str, default = 'MILP')

    # Jupiter
    parser.add_argument('--spread', type=float, default = 0.5)

    # COPE
    parser.add_argument('--beta', type=float, default = 1.5)
    parser.add_argument('--budget', action='store_true', default=False)
    parser.add_argument('--type', type=str, default="dense")
    parser.add_argument('--max_ports', type=int, default=10,
                      help='Maximum number of ports per node')
    parser.add_argument('--base_capacity', type=int, default=3000,
                      help='Base capacity for network links')
    parser.add_argument('--mode', type=str, default = 'test')
    return parser

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    
    return parser.parse_args(args)