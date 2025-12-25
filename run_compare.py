#!/usr/bin/env python3
import sys
import os
from typing import List

# Ensure project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from ToTE_benchmarks.ToTE_helper import parse_args

# Import benchmark entry points
from ToTE_benchmarks.ATRO import benchmark as atro_bench
from ToTE_benchmarks.COUDER import benchmark as couder_bench
from ToTE_benchmarks.MILP import benchmark as milp_bench

import importlib.util
from types import ModuleType

SUPPORTED_TOTE = {
    'ATRO': atro_bench,
    'COUDER': couder_bench,
    'MILP': milp_bench,
}

SUPPORTED_TO = {
    'ABSM': os.path.join(ROOT, 'To-benchmarks', 'ABSM.py'),
    'BG': os.path.join(ROOT, 'To-benchmarks', 'BG.py'),
    'MILPD': os.path.join(ROOT, 'To-benchmarks', 'MILP_d.py'),
    'MMCF': os.path.join(ROOT, 'To-benchmarks', 'mmcf.py'),
}

USAGE = """
Run comparisons across methods.

Examples:
  # Compare TOTE methods on Facebook_pod_a
    python run_compare.py tote --methods ATRO,COUDER,MILP --topo_name Facebook_pod_a --mode test

  # Compare To-benchmarks on Topo_16
  python run_compare.py to --methods ABSM,BG,MILPD,MMCF --topo_name Topo_16 --mode test

Notes:
- MILP/MILPD may require Gurobi; ATRO/COUDER/MMCF/ABSM rely on OR-Tools and NumPy.
- Results are written under Result/<topo>/<METHOD>/*.txt by each benchmark.
"""


def parse_methods(csv: str) -> List[str]:
    return [m.strip().upper() for m in csv.split(',') if m.strip()]


def run_group(group: str, methods: List[str], argv: List[str]):
    # Parse common props once using existing helper
    props = parse_args(argv)
    # Choose registry
    if group == 'tote':
        registry = SUPPORTED_TOTE
    elif group == 'to':
        registry = SUPPORTED_TO
    else:
        print('Unknown group:', group)
        print(USAGE)
        sys.exit(1)

    # Default to all methods in the group
    if not methods:
        methods = list(registry.keys())

    print(f"Running group={group} methods={methods} topo={props.topo_name} mode={props.mode}")

    # Run each method
    for method in methods:
        entry = registry.get(method)
        if entry is None:
            print(f"Skip unknown method: {method}")
            continue
        # Resolve benchmark function
        if group == 'tote':
            bench = entry
        else:
            # Dynamic import by file path for 'To-benchmarks' (hyphenated dir)
            mod_path = entry
            if not os.path.exists(mod_path):
                print(f"[WARN] {method} script not found: {mod_path}")
                continue
            mod_name = f"_to_bench_{method.lower()}"
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            if spec is None or spec.loader is None:
                print(f"[WARN] Cannot load module for {method} from {mod_path}")
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[ERROR] Loading {method} failed: {e}")
                continue
            if not hasattr(module, 'benchmark'):
                print(f"[WARN] No benchmark(props) in {mod_path}")
                continue
            bench = getattr(module, 'benchmark')

        # Set solver name into props for downstream scripts
        props.TE_solver = method
        print(f"\n=== Running {method} ===")
        try:
            bench(props)
        except ModuleNotFoundError as e:
            print(f"[WARN] {method} missing dependency: {e}")
        except Exception as e:
            print(f"[ERROR] {method} failed: {e}")


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    group = sys.argv[1].lower()
    # Extract optional --methods CSV if present, pass the rest to parse_args
    args = sys.argv[2:]
    methods = []
    if '--methods' in args:
        i = args.index('--methods')
        if i + 1 < len(args):
            methods = parse_methods(args[i + 1])
            # Remove from argv forwarded to parse_args
            del args[i:i+2]
    run_group(group, methods, args)


if __name__ == '__main__':
    main()
