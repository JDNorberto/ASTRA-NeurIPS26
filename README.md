
# ASTRA-NeurIPS26

This repository contains the code for the paper **"ASTRA: ADMM-Accelerated Topology Reconfiguration for Dynamic Satellite Constellations."**

## Dependencies

The following packages are required to run the code:

- `numpy==2.4.3`
- `cvxpy==1.8.1`
- `networkx==3.6.1`
- `scipy==1.17.1`

## Repository Structure

The code is organized as follows:

### `code/`

The `code` directory should be marked as a **source directory**.

In PyCharm, go to **Settings → Project Structure**, select the `code` directory, and mark it as a source directory.

- **`constellation_info/`** — Contains the information necessary to run some experiments on Starlink data.

- **`pickle_files/`** — Contains pickled files to make plotting faster.

- **`algorithms/`** — Contains the implementations of GUTO and ASTRA.
  - `admm_topology_configuration.py` — Contains both the online and offline ASTRA algorithms.
  - `offline_topology_configuration/` — Contains GUTO.
  - `sota.py` — Contains the implementations of +Grid and xGrid.
  - `tc_utils.py` — Contains various utility functions for topology configuration.

- **`graph_analysis.py`** — Contains all graph-related methods.

- **`satellite_simulation.py`** — Used to generate an Iridium-like constellation.

- **`utility_functions.py`** — Contains all implemented utility functions.

- **`algorithm_evaluation.py`** — Contains the code used for the main experiments presented in the paper.

- **`main.py`** — Runs the proposed algorithms using user-specified positions and parameters.

## Running the Algorithms

If you want to run the code to replicate the experiments of the paper, you should refer to **`algorithm_evaluation.py`**. In case you want to test different parameters or different constellations, the `main.py` script should be called as follows:

```bash
python main.py --cf <config-file> --pf <position-file>
```

The position file should contain normalized positions between 0 and 1. When using **online algorithms**, two additional arguments are required:

- `--u <utility-file>` — Specifies the file containing all precomputed utility matrices.
- `--fs <first-solution>` — Specifies the initial topology to be considered.

The provided `config_file.json` should be used to specify the algorithm parameters.
