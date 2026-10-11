import argparse
import inspect
import json
import pickle
import uuid
from operator import itemgetter
import numpy as np
from algorithms.offline_topology_configuration import laplacian_matrix_with_utilities
from algorithms.admm_topology_configuration import admm_online_gradient_descent, admm_online_conditional_gradient, \
                                                    admm_offline
from graph_analysis import create_graph

_CONFIG_FILE_HELP_STR = "Config file used to select the algorithm to run and the values of its parameters."
_POSITION_FILE_HELP_STR = ("Pickle or Numpy txt file with the position of the satellites. If the online algorithms are being used,"
                           "it is expected that this file provides an array of positions.")
_UTILITY_FILE_HELP_STR = "Pickle file containing an array of utility matrices to be used by online algorithms."
_FS_FILE_HELP_STR = "Pickle file with a satellite network topology."

_STR_TO_OFF_ALG = {
    "GUTO": laplacian_matrix_with_utilities,
    "ASTRA": admm_offline,
}

_STR_TO_ON_ALG = {
    "ASTRA-OGD": admm_online_gradient_descent,
    "ASTRA-OCG": admm_online_conditional_gradient,
}

def _print_curr_config(alg_name, params, value):
    f_str = 5*"-"+ f" Current Configuration for {alg_name} "+ 5*"-"
    print(f_str)
    for p, v in zip(params, value):
        print(f"Parameter {p} with value = {v}")
    print(len(f_str)*"-")


def main_on(alg_name, config, pos, ut_file, fs_file):
    if ut_file is None:
        raise RuntimeError("Utility file is required")
    if fs_file is None:
        raise RuntimeError("First Solution file is required")

    ut_mats = pickle.load(open(ut_file, "rb"))
    X = pickle.load(open(fs_file, "rb"))

    alg = _STR_TO_ON_ALG[alg_name]
    alg_params = inspect.getfullargspec(alg)[0]
    params = config["parameters"]
    mask_mat = np.identity(len(pos[0]))
    runs = len(pos)

    first_args = itemgetter(alg_params[1])(params)
    last_args = itemgetter(*alg_params[6:])(params)

    results = alg(runs, first_args, X, ut_mats, mask_mat, [pos], *last_args)
    return results


def main_off(alg_name, config, pos):
    alg = _STR_TO_OFF_ALG[alg_name]
    alg_params = inspect.getfullargspec(alg)[0]
    params = config["parameters"]
    params["sats_per_plane"] = np.array(params["sats_per_plane"], dtype=np.int32)
    fun_args = itemgetter(*alg_params[1:])(params)
    _print_curr_config(alg_name, alg_params[1:], fun_args)

    res_mat = alg(pos, *fun_args)
    # _, _ = create_graph(res_mat, params["max_isl"], len(pos)-params["num_bs"], params["num_bs"], params["tau"])

    return res_mat


def main(config_file, pos_file, ut_file, fs_file):

    run_uuid = str(uuid.uuid4())

    with open(config_file, "r") as cf:
        config = json.load(cf)
        alg_name = config["algorithm"].upper()

        file_ext = pos_file.split(".")[-1]
        if file_ext == "pkl" or file_ext == "pickle":
            pos = pickle.load(open(pos_file, "rb"))
            pos = np.array(pos, dtype=np.float32)
        elif file_ext == "txt":
            pos = np.loadtxt(pos_file, dtype=np.float32)
        else:
            raise RuntimeError(f"Unknown extension {file_ext}")

        if alg_name in _STR_TO_OFF_ALG:
            res = main_off(alg_name, config, pos)
        else:
            res = main_on(alg_name, config, pos, ut_file, fs_file)

    pickle.dump(res, open("./output/" + alg_name + "_" + run_uuid + ".pickle", "wb"))
    with open(f"./output/config_{run_uuid}", "w") as f:
        f.write(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', "--config-file", type=str, required=True, help=_CONFIG_FILE_HELP_STR)
    parser.add_argument('--pf', "--pos-file", type=str, required=True, help=_POSITION_FILE_HELP_STR)
    parser.add_argument('--u', "--utility", type=str, required=False, help=_UTILITY_FILE_HELP_STR)
    parser.add_argument('--fs', "--first-solution", type=str, required=False, help=_FS_FILE_HELP_STR)
    args = parser.parse_args()
    main(config_file=args.cf, pos_file=args.pf, ut_file=args.u, fs_file=args.fs)