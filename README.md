# ASTRA-NeurIPS26

This repo contains the code for the paper "ASTRA: ADMM-Accelerated Topology Reconfiguration for Dynamic Satellite Constellations".

The packages used to run this code are: <br>

- numpy 2.4.3

- cvxpy 1.8.1

- networkx 3.6.1

- scipy 1.17.1

The code is divided as follows:

- code, should be marked as a source directory (PyCharm: Settings -> Project Structure -> select code dir and mark as source).

  - constellation_info, contains the information necessary to run some experiments on Starlink data.

  - pickle_files, contains pickled files to make plots quicker.

  - algorithms, contains the code for GUTO and ASTRA.

    - admm_topology_configuration.py, contains both Online and Offline ASTRA algorithms.

    - offline_topology_configuration, contains GUTO.

    - sota.py, contains the implementations for +Grid and xGrid.

    - tc_utils.py, contains a plethora of methods that are useful for topology configuration.

  - algorithm_evaluation.py, the main file where experiments can be executed.

  - graph_analysis.py, contains all graph-related methods.

  - satellite_simulation.py, used to generate an Iridium-like constellation.

  - utility_functions.py, contains all the implemented utility functions.
