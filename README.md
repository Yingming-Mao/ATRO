


# ATRO: A Fast Algorithm for Topology Engineering of Reconfigurable DCNs

[![Conference](https://img.shields.io/badge/INFOCOM%202025-Accepted-b31b1b.svg)](https://infocom.ieee.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18054616.svg)](https://doi.org/10.5281/zenodo.18054616)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> **Note:** This is the official code release for the paper **"ATRO: A Fast Algorithm for Topology Engineering of Reconfigurable Datacenter Networks"**, accepted at **IEEE INFOCOM 2025**.

## 📖 Overview

Reconfigurable Datacenter Networks (RDCNs) promise flexible bandwidth allocation but often suffer from slow reconfiguration speeds. **ATRO** ( Alternating Topology and Routing Optimization) addresses this challenge by providing a high-speed topology engineering framework.

This repository contains a **minimal, runnable subset** of the ATRO algorithms, baselines, and necessary datasets to reproduce key experiments. It focuses on the core logic:

* **ATRO**: The proposed fast topology engineering algorithm.
* **ABSM**: The single-hop version of ATRO (included in classic benchmarks).
* **Baselines**: Implementations of COUDER, MILP, BG, MILPD, and MMCF.

---

## 📂 Repository Layout

To keep the footprint lightweight, visualization assets and raw topology generation scripts have been excluded.

```text
├── Data/                 
│   └── Facebook_pod_a/   # Trimmed dataset: topology, tunnels, and test traces
├── src/                  # Core configurations and utility functions
├── TO_benchmarks/        # One-hop Topology Engineering (ABSM, MCF, etc.)
├── ToTE_benchmarks/      # ToTE methods (ATRO, COUDER) and helpers
├── run_demo.py           # Entry point for single-run demos
├── run_compare.py        # Entry point for batched comparisons
├── requirements.txt      # Minimal dependencies
└── environment.yml       # Conda environment configuration
```


## 📜 License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for the full terms.
© 2025 ATRO Authors.

------

## 🛠️ Requirements

- **Python 3.8+**
- **Core Libs:** `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`
- **Solvers:**
  - [Google OR-Tools](https://developers.google.com/optimization) (Required for COUDER/MMCF)
  - [Gurobi](https://www.gurobi.com/) (Optional, but **required** for LP-based baselines like MILP/MILPD). *Academic licenses are free.*

### Installation

**Option 1: Conda (Recommended)**

Bash

```
conda env create -f environment.yml
conda activate ATRO
```

**Option 2: Pip**

Bash

```
pip install -r requirements.txt
```

------

## 🚀 Quick Start

### 1. Run ATRO Demo

Run the proposed ATRO algorithm on the `Facebook_pod_a` dataset:

Bash

```
python run_demo.py atro --topo_name Facebook_pod_a --mode test --max_ports 10
```

> **Output:** Results (metrics and runtime) will be saved to `Result/Facebook_pod_a/ATRO/`.

### 2. Batch Comparisons (Two-Hops)

Compare ATRO against state-of-the-art ToTE methods (COUDER, MILP):

Bash

```
python run_compare.py tote \
    --methods ATRO,COUDER,MILP \
    --topo_name Facebook_pod_a \
    --mode test \
    --max_ports 10
```

### 3. Batch Comparisons (One-Hop)

Compare against one-hop case (ABSM, BG, MILPD, MMCF).

Note: ABSM is the one-hop version of ATRO.

Bash

```
python run_compare.py to \
    --methods ABSM,BG,MILPD,MMCF \
    --topo_name Facebook_pod_a \
    --mode test \
    --max_ports 10
```

------

## ⚙️ Arguments & Configuration

Common flags managed by `ToTE_benchmarks/ToTE_helper.py`:

| **Argument**      | **Default**      | **Description**                       |
| ----------------- | ---------------- | ------------------------------------- |
| `--topo_name`     | `Facebook_pod_a` | Dataset folder name under `Data/`     |
| `--paths_file`    | `tunnels.txt`    | Candidate path file name              |
| `--max_ports`     | `10`             | Maximum reconfigurable ports per node |
| `--base_capacity` | `3000`           | Base link capacity                    |
| `--mode`          | `test`           | Run mode: `train` or `test`           |

For `run_compare.py`, you can also use `--methods` (comma-separated) to select specific algorithms.

------

## ⚠️ Notes

1. **Gurobi License:** Ensure you have a valid Gurobi license configured if you intend to run `MILP` or `MILPD` baselines.
2. **Intermediate Files:** The code generates intermediate artifacts (e.g., optimal values) in the `Data/` directory during execution.
3. **Custom Data:** To add new topologies, create a folder under `Data/<your_topo>/` following the structure of `Facebook_pod_a`.

------

## 📝 Citation

If you find this code or our paper useful in your research, please cite our **INFOCOM 2025** paper:

```
@misc{mao2025atrofastalgorithmtopology,
      title={ATRO: A Fast Algorithm for Topology Engineering of Reconfigurable Datacenter Networks}, 
      author={Yingming Mao and Qiaozhu Zhai and Ximeng Liu and Xinchi Han and Fafan li and Shizhen Zhao and Yuzhou Zhou and Zhen Yao and Xia Zhu},
      year={2025},
      eprint={2507.13717},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2507.13717}, 
}
```

*For the preprint version, you may also refer to [arXiv:2507.13717](https://arxiv.org/abs/2507.13717).*
