#!/usr/bin/env python3
import sys
import os

# Ensure project root on path
cur = os.path.dirname(os.path.abspath(__file__))
root = cur
sys.path.append(root)

from ToTE_benchmarks.ToTE_helper import parse_args
from ToTE_benchmarks.ATRO import benchmark as atro_benchmark

USAGE = """
Usage examples:
    python run_demo.py atro --topo_name Facebook_pod_a
    python run_demo.py atro --topo_name Topo_16

Notes:
- Supported topologies: Facebook_pod_a, Topo_16 (under Data/).
- Some benchmarks may produce plots; ensure matplotlib is installed (included in requirements).
"""


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    which = sys.argv[1].lower()
    args = sys.argv[2:]
    props = parse_args(args)

    if which == 'atro':
        # Ensure correct solver is selected for ATRO base run
        props.TE_solver = 'ATRO'
        atro_benchmark(props)
    else:
        print("Unknown benchmark:", which)
        print(USAGE)
        sys.exit(1)


if __name__ == '__main__':
    main()
