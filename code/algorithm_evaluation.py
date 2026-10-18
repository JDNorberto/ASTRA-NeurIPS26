import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pickle
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from algorithms.admm_topology_configuration import admm_online_conditional_gradient, \
    admm_online_gradient_descent, admm_offline
from algorithms.offline_topology_configuration import laplacian_matrix_with_utilities
from algorithms.sota import plus_grid, x_grid
from algorithms.tc_utils import create_connectivity_incentive, create_diag_mask
from graph_analysis import create_graph, graph_statistics
from satellite_simulation import generate_constellation, move_constellation
from utility_functions import CvxUtility


def _get_sats_idxs():
    orb_plane_sat = np.loadtxt("./constellation_info/orbital_info.txt")
    orb_plane_sat = np.array(sorted(orb_plane_sat, key=lambda x: x[0]), dtype=np.int32)
    _, sats_per_plane = np.unique(orb_plane_sat[:, 0], return_counts=True)
    idxs = orb_plane_sat[:, 1]

    # Remove planes that have a few satellite
    planes_to_remove = np.where(sats_per_plane <= 5)[0]
    sats_per_plane = np.delete(sats_per_plane, planes_to_remove)
    print(sats_per_plane.min(), sats_per_plane.max())
    sats_to_remove = []
    for plane, sat in orb_plane_sat:
        if plane in planes_to_remove:
            sats_to_remove.append(sat)

    return np.delete(idxs, sats_to_remove), sats_per_plane


def evaluate_on_iridium():
    pos = generate_constellation(0.85, 0.75, 6, 11).reshape(-1, 3)
    sats_per_plane = np.repeat(11, 6)
    print(len(pos), len(sats_per_plane))
    max_dist = 0.8

    s = time.time()
    mat, _, it = laplacian_matrix_with_utilities(pos, [0.6, 0.4], ["long_dist", "sat_dist"],
                                                 4, max_dist, sats_per_plane, len(sats_per_plane), 0,
                                                 cone=True, verbose=False)
    print("Overall time: ", time.time() - s)
    print("Num its: ", it[0])
    print("Solve time: ", it[1])
    print("Preprocessing time: ", it[2])

    _, _ = create_graph(mat, 4, len(pos), 0, threshold=-0.5, print_metrics=True)

    s = time.time()
    mat = admm_offline(pos, 0, 1, 1, 1e-4, 4, 0, max_dist, 5000,
                       ["long_dist", "sat_dist"], sats_per_plane, len(sats_per_plane), cone=True,
                       weights=[0.6, 0.4], projected_its=5, eps=1e-7)
    print("ADMM time: ", time.time() - s)
    _, _ = create_graph(mat, 4, len(pos), 0, threshold=-0.5, print_metrics=True)


def evaluate_on_starlink_data(max_dist=0.8, max_isls=4, delta_time=1):
    idxs, sats_per_plane = _get_sats_idxs()

    off_metrics = []
    times = []
    prepro_time = []
    solve_time = []
    num_its = []

    admm_metrics = []
    admm_times = []

    plus_grid_metrics = []
    plus_grid_times = []

    cross_grid_metrics = []
    cross_grid_times = []

    for i in range(0, 4, delta_time):
        file_path = f"./constellation_info/const_state_{(i+3)*5}.txt"
        print(f"Evaluating file {file_path}", flush=True)
        pos = np.loadtxt(file_path, dtype=np.float32)
        pos = pos[idxs]

        #CVXPY Algorithm
        start = time.time()
        mat, _, its, P = laplacian_matrix_with_utilities(pos, [0.6, 0.4], ["long_dist", "sat_dist"],
                                                      max_isls, max_dist, sats_per_plane, len(sats_per_plane),
                                                      0, cone=False, verbose=False)

        times.append(time.time() - start)
        num_its.append(its[0])
        solve_time.append(its[1])
        prepro_time.append(its[2])
        _, metrics = create_graph(mat, max_isls, len(pos), 0, threshold=-0.45, print_metrics=False)
        off_metrics.append(metrics)

        #ADMM Algorithm
        start = time.time()
        mat = admm_offline(pos, 0, 1, 1, 1e-4, 4, 0, max_dist, 10000,
                           ["long_dist", "sat_dist"], sats_per_plane, len(sats_per_plane),
                           projected_its=20, cone=False, eps=1e-7, weights=[0.6, 0.4])
        admm_times.append(time.time() - start)
        _, metrics = create_graph(mat, max_isls, len(pos), 0, threshold=-0.45, print_metrics=False)
        admm_metrics.append(metrics)

        #+Grid Algorithm
        start = time.time()
        edge_list = plus_grid(pos, sats_per_plane)
        plus_grid_times.append(time.time() - start)
        metrics = graph_statistics(nx.Graph(edge_list), print_metrics=False)
        plus_grid_metrics.append(metrics)

        #xGrid Algorithm
        start = time.time()
        edge_list = x_grid(pos, sats_per_plane)
        cross_grid_times.append(time.time() - start)
        metrics = graph_statistics(nx.Graph(edge_list), print_metrics=False)
        cross_grid_metrics.append(metrics)

    print(10 * "-", "Graph Metrics:", 10 * "-")
    print("Mean metrics: ", list(map(lambda x: round(x, 3), np.mean(off_metrics, axis=0))))
    print("Std metrics: ", list(map(lambda x: round(x, 3), np.std(off_metrics, axis=0))))
    print(10 * "-", "Method Time:", 10 * "-")
    print("Mean its: ", np.mean(num_its))
    print("Std its: ", np.std(num_its))
    print("---------------------")
    print("Mean preprocessing time: ", np.mean(prepro_time))
    print("Std preprocessing time: ", np.std(prepro_time))
    print("---------------------")
    print("Mean solve time: ", np.mean(solve_time))
    print("Std solve time: ", np.std(solve_time))
    print("---------------------")
    print("Mean time: ", np.mean(times))
    print("Std time: ", np.std(times))
    print("---------------------\n")
    print(10 * "-", "ADMM Graph Metrics:", 10 * "-")
    print("Mean metrics: ", list(map(lambda x: round(x, 3), np.mean(admm_metrics, axis=0))))
    print("Std metrics: ", list(map(lambda x: round(x, 3), np.std(admm_metrics, axis=0))))
    print("---------------------")
    print("Mean time: ", np.mean(admm_times))
    print("Std time: ", np.std(admm_times))
    print("---------------------\n")
    print(10 * "-", "+Grid Graph Metrics:", 10 * "-")
    print("Mean metrics: ", list(map(lambda x: round(x, 3), np.mean(plus_grid_metrics, axis=0))))
    print("Std metrics: ", list(map(lambda x: round(x, 3), np.std(plus_grid_metrics, axis=0))))
    print("---------------------")
    print("Mean time: ", np.mean(plus_grid_times))
    print("Std time: ", np.std(plus_grid_times))
    print("---------------------\n")
    print(10 * "-", "xGrid Graph Metrics:", 10 * "-")
    print("Mean metrics: ", list(map(lambda x: round(x, 3), np.mean(cross_grid_metrics, axis=0))))
    print("Std metrics: ", list(map(lambda x: round(x, 3), np.std(cross_grid_metrics, axis=0))))
    print("---------------------")
    print("Mean time: ", np.mean(cross_grid_times))
    print("Std time: ", np.std(cross_grid_times))
    print("---------------------\n")


def evaluate_online_algorithms_starlink(max_dist=0.8, max_isls=4, delta_time=5):
    idxs, sats_per_plane = _get_sats_idxs()

    positions = []
    for i in range(5, 20, delta_time):
        file_path = f"./constellation_info/const_state_{i}.txt"
        print(f"Evaluating file {file_path}", flush=True)
        pos = np.loadtxt(file_path, dtype=np.float32)
        pos = pos[idxs]
        positions.append(pos)

    pos0 = np.loadtxt("./constellation_info/const_state_0.txt", dtype=np.float32)
    pos0 = pos0[idxs]
    init_mat = laplacian_matrix_with_utilities(pos0, [0.6, 0.4], ["long_dist", "sat_dist"], max_isls,
                                               max_dist, sats_per_plane, len(sats_per_plane),
                                               0, cone=False)
    init_mat = np.array(init_mat, dtype=np.float32)

    runs = len(positions)
    mask_mat = np.diag(np.ones(len(init_mat)))
    num_bs = 0

    utility_mask = []
    for pos in positions:
        cvx_ut = CvxUtility(["long_dist", "sat_dist"], [0.6, 0.4], len(pos), sats_pos=pos,
                        sats_per_plane=sats_per_plane, k=3, num_planes=len(sats_per_plane), num_bs=num_bs)
        utility_mask.append(cvx_ut())

    print("Starting Online Algorithms")
    s = time.time()
    admm_ogd_sols = admm_online_gradient_descent(runs, 100_000, init_mat, utility_mask, mask_mat,
                                                 positions, num_bs, max_isls, 0, max_dist, 1,
                                                 lr=np.sqrt(12490)/np.sqrt(runs), cone=False)
    print("Admm Online Gradient Descent time: ", (time.time() - s)/runs)

    s = time.time()
    admm_ocg_sols = admm_online_conditional_gradient(runs, 100_000, init_mat, utility_mask, mask_mat, positions,
                                                     num_bs, max_isls, 0, max_dist, 1, cone=False)
    print("Admm Online Conditional Gradient time: ", (time.time() - s)/runs)

    _get_online_metrics(admm_ogd_sols, admm_ocg_sols, max_isls, num_sat=len(positions[0])-num_bs, num_bs=num_bs)


def _get_online_metrics(admm_ogd_sols, admm_ocg_sols, max_isls, num_sat, num_bs):
    admm_ocg = []
    admm_ogd = []
    for admm_ogd_mat, admm_ocg_mat in zip(admm_ogd_sols, admm_ocg_sols):
        _, admm_ogd_m = create_graph(admm_ogd_mat, max_isls, num_sat, num_bs, threshold=-0.45, print_metrics=False)
        _, admm_ocg_m = create_graph(admm_ocg_mat, max_isls, num_sat, num_bs, threshold=-0.45, print_metrics=False)

        admm_ocg.append(admm_ocg_m)
        admm_ogd.append(admm_ogd_m)

    print(10 * "-", "ADMM-OCG Graph Metrics:", 10 * "-")
    print("Mean metrics: ", list(map(lambda x: round(x, 3), np.mean(admm_ocg, axis=0))))
    print("Std metrics: ", list(map(lambda x: round(x, 3), np.std(admm_ocg, axis=0))))
    print(10 * "-", "ADMM-OGD Graph Metrics:", 10 * "-")
    print("Mean metrics: ", list(map(lambda x: round(x, 3), np.mean(admm_ogd, axis=0))))
    print("Std metrics: ", list(map(lambda x: round(x, 3), np.std(admm_ogd, axis=0))))


def ablation_lambda_app():
    pos = generate_constellation(0.85, 0.75, 6, 11).reshape(-1, 3)
    sats_per_plane = np.repeat(11, 6)
    t = -0.5
    max_dist = 0.8
    weight_list = [[0, 1], [0.15, 0.85], [0.30, 0.70], [0.45, 0.55],
                   [0.60, 0.40], [0.75, 0.25], [0.90, 0.10], [1, 0]]
    for w in weight_list:
        print("Current weights: ", w)
        s = time.time()
        mat = admm_offline(pos, 0, 1, 1, 1e-4, 4, 0, max_dist, 500,
                           ["long_dist", "sat_dist"], sats_per_plane, len(sats_per_plane),
                           weights=w, cone=False, eps=1e-7, projected_its=20)
        print("Time: ", time.time() - s)
        _, _ = create_graph(mat, 4, len(pos), 0, threshold=t, print_metrics=True)
        print("\n")


def ablation_rho_app():
    pos = generate_constellation(0.85, 0.75, 6, 11).reshape(-1, 3)
    sats_per_plane = np.repeat(11, 6)
    weights = [0.6, 0.4]
    t = -0.5

    max_dist = 0.8
    rho_list = [4, 2, 1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001]
    for rho in rho_list:
        print("Current rho: ", rho, "/ Current weights: ", weights)
        s = time.time()
        mat = admm_offline(pos, 0, rho, 1, 1e-4, 4, 0, max_dist, 500,
                           ["long_dist", "sat_dist"], sats_per_plane, len(sats_per_plane),
                           weights=weights, cone=True, eps=1e-7, projected_its=20)
        print("Time: ", time.time() - s)
        e, _ = create_graph(mat, 4, len(pos), 0, threshold=t, print_metrics=True)


def ablation_projected_its(p_its_list):
    pos = generate_constellation(0.85, 0.75, 6, 11).reshape(-1, 3)
    sats_per_plane = np.repeat(11, 6)
    weights = [0.6, 0.4]
    t = -0.5
    rho = 1

    max_dist = 0.8
    for p_its in p_its_list:
        print("Current projected iterations: ", p_its)
        s = time.time()
        mat = admm_offline(pos, 0, rho, 1, 1e-4, 4, 0, max_dist, 500,
                           ["long_dist", "sat_dist"], sats_per_plane, len(sats_per_plane),
                           weights=weights, cone=False, projected_its=p_its, eps=1e-7)
        print(time.time() - s)
        _, _ = create_graph(mat, 4, len(pos), 0, threshold=t, print_metrics=True)
        print("\n")


def get_times(runs=10):
    idxs, sats_per_plane = _get_sats_idxs()

    pos = np.loadtxt("./constellation_info/const_state_0.txt")
    pos = pos[idxs]

    times = []
    for _ in range(runs):
        s = time.time()
        plus_grid(pos, sats_per_plane)
        times.append(time.time() - s)
    print("+Grid mean times: ", np.mean(times))
    print("+Grid std times: ", np.std(times))

    times = []
    for _ in range(runs):
        s = time.time()
        x_grid(pos, sats_per_plane)
        times.append(time.time() - s)
    print("xGrid mean times: ", np.mean(times))
    print("xGrid std times: ", np.std(times))

    rho = 1
    weights = [0.6, 0.4]
    max_dist = 0.8
    epsilon = 1e-7
    times = []

    for i in range(runs):
        print("Iteration: ", i)
        s = time.time()
        admm_offline(pos, 0, rho, 1, 1e-4, 4, 0, max_dist, 10000,
                     ["long_dist", "sat_dist"], sats_per_plane, len(sats_per_plane),
                     weights=weights, cone=False, projected_its=20, eps=epsilon)
        times.append(time.time() - s)
        print(times[i])
    print("ASTRA mean times: ", np.mean(times))
    print("ASTRA std times: ", np.std(times))

    times = []
    for i in range(runs):
        print("Iteration: ", i)
        s = time.time()
        laplacian_matrix_with_utilities(pos, weights, ["long_dist", "sat_dist"], 4,
                                        max_dist, sats_per_plane, len(sats_per_plane), 0,
                                        cone=False, verbose=False)
        times.append(time.time() - s)
        print(times[i])
    print("GUTO mean times: ", np.mean(times))
    print("GUTO std times: ", np.std(times))


def loss_function(X, P, U, M):
    loss_1 = 0.5 * np.linalg.norm(np.multiply((X - P), U), "fro") ** 2
    loss_2 = np.trace(np.multiply(X, M))
    return loss_1 - loss_2


def astra_get_true_preds_and_positions(runs, a, b, cone=False):
    sats_pos = generate_constellation(a, b, 6, 11).reshape(-1, 3)
    funs_to_use = ["long_dist", "sat_dist"]
    new_weights = [0.6, 0.4]
    X_one = admm_offline(sats_pos, 0, 1, 1, 1e-4, 4, 0, 0.8,
                         5000, funs_to_use, sats_per_plane=np.repeat(11, 6), num_planes=6,
                         weights=new_weights, eps=1e-7, cone=cone, projected_its=5)
    true_X_values = []
    positions = []
    for i in range(runs):
        sats_pos = move_constellation(sats_pos, 11, 6).reshape(-1, 3)
        positions.append(sats_pos)
        X = admm_offline(sats_pos, 0, 1, 1, 1e-4, 4, 0, 0.8,
                             5000, funs_to_use, sats_per_plane=np.repeat(11, 6), num_planes=6,
                             weights=new_weights, eps=1e-7, cone=cone, projected_its=5)
        true_X_values.append(X)

    return X_one, true_X_values, positions


def plot_regret_curve(runs):
    X_one, true_X_values, positions = astra_get_true_preds_and_positions(runs, .85, .75, cone=False)

    utility_mask = []
    Ps = []
    for p in positions:
        P = create_connectivity_incentive(p, 0, 0.8, cone=False)
        Ps.append(P)

        cvx_ut = CvxUtility(["long_dist", "sat_dist"], [0.6, 0.4], len(p),
                            sats_per_plane=np.repeat(11, 6), k=3, num_planes=6, num_bs=0,
                            sats_pos=p)
        utility_mask.append(cvx_ut())
    mask_mat = np.diag(np.full(len(X_one), 1e-4))

    comp_loss = []
    for i in range(len(positions)):
        X = true_X_values[i]
        P = Ps[i]
        U = utility_mask[i]
        comp_loss.append(loss_function(X, P, U, mask_mat))
    comp_loss = np.cumsum(comp_loss)

    admm_sols_static = admm_online_gradient_descent(runs, 5000, X_one, utility_mask,
                                                    np.diag(np.full(len(X_one), 1)), positions, 0,
                                                    4, 0, 0.8, 1, lr=1 / np.sqrt(runs))
    admm_sols_dynamic = admm_online_gradient_descent(runs, 5000, X_one, utility_mask,
                                                     np.diag(np.full(len(X_one), 1)), positions, 0,
                                                     4, 0, 0.8, 1, lr=np.sqrt(9393) / np.sqrt(runs))

    stat_loss = []
    dyn_loss = []
    for i in range(len(positions)):
        P = Ps[i]
        U = utility_mask[i]
        stat_loss.append(loss_function(admm_sols_static[i], P, U, mask_mat))
        dyn_loss.append(loss_function(admm_sols_dynamic[i], P, U, mask_mat))
    stat_loss = np.cumsum(stat_loss)
    dyn_loss = np.cumsum(dyn_loss)

    regret_static = []
    regret_dynamic = []
    for i in range(runs):
        regret_static.append(stat_loss[i] - comp_loss[i])
        regret_dynamic.append(dyn_loss[i] - comp_loss[i])

    FSIZE = 24
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(np.arange(0, 1000, 1), regret_static, label=r"ASTRA-OGD with step size $\frac{1}{\sqrt{T}}$")
    ax.plot(np.arange(0, 1000, 1), regret_dynamic, label=r"ASTRA-OGD with step size $\frac{\sqrt{9393}}{\sqrt{T}}$")
    ax.set_xlabel("Number of Iterations", fontsize=FSIZE)
    ax.set_ylabel("Regret", fontsize=FSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=FSIZE)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=FSIZE - 2)
    plt.tight_layout()
    plt.savefig("step_size_regret_comparison.pdf")
    plt.show()


def calculate_avg_rpe(mat_array, true_X_values):
    rpe = []
    for i in range(len(true_X_values)):
        admm_norm = np.linalg.norm(true_X_values[i] - mat_array[i]) ** 2

        num_entries = len(np.where(true_X_values[i] != 0)[0])

        rpe.append(admm_norm / num_entries)
    return rpe


def plot_multiple_seed_sat_rem(n_runs):
    ogd_rpes = []
    ocg_rpes = []
    seeds = [42, 43, 44]
    for s in seeds:
        ress_ocg = pickle.load(open(f"./pickle_files/seed{s}_ocg_sols_sats_rem.pickle", "rb"))
        ress_ogd = pickle.load(open(f"./pickle_files/seed{s}_ogd_sols_sats_rem.pickle", "rb"))
        true_X_values = pickle.load(open(f"./pickle_files/seed{s}_true_X_values_sats_rem.pickle", "rb"))
        ress_rpe_ocg = calculate_avg_rpe(ress_ocg, true_X_values)
        ocg_rpes.append(ress_rpe_ocg)
        ress_rpe_ogd = calculate_avg_rpe(ress_ogd, true_X_values)
        ogd_rpes.append(ress_rpe_ogd)

    ogd_reps = np.array(ogd_rpes).reshape(n_runs, -1)
    ocg_reps = np.array(ocg_rpes).reshape(n_runs, -1)

    ogd_reps_mean = np.array([0, *gaussian_filter1d(np.mean(ogd_reps, axis=0), sigma=.65)])
    ogd_reps_std = np.array([0, *gaussian_filter1d(np.std(ogd_reps, axis=0), sigma=.65)])
    ocg_reps_mean = np.array([0, *gaussian_filter1d(np.mean(ocg_reps, axis=0), sigma=.65)])
    ocg_reps_std = np.array([0, *gaussian_filter1d(np.std(ocg_reps, axis=0), sigma=.65)])

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(np.arange(0, len(ogd_reps_mean), 1),
            ogd_reps_mean,
            label='ASTRA-OGD', c='purple')
    ax.plot(np.arange(0, len(ocg_reps_mean), 1),
            ocg_reps_mean,
            label='ASTRA-OCG', c='darkgreen')

    # Plot ±std bands
    ax.fill_between(
        np.arange(0, len(ogd_reps_mean)),
        ogd_reps_mean - ogd_reps_std,
        ogd_reps_mean + ogd_reps_std,
        color='purple',
        alpha=0.2
    )

    ax.fill_between(
        np.arange(0, len(ocg_reps_mean)),
        ocg_reps_mean - ocg_reps_std,
        ocg_reps_mean + ocg_reps_std,
        color='green',
        alpha=0.2
    )

    FSIZE = 24
    TSIZE = 22
    plt.axvline(x=301, color='grey', linestyle='-.', linewidth=2, label='Satellite Removal')
    plt.axvline(x=650, color='grey', linestyle='--', linewidth=2, label='Satellite Addition')
    plt.xlabel("Number of Iterations", fontsize=FSIZE)
    plt.xticks(np.arange(0, len(ogd_reps_mean), 100), fontsize=TSIZE)
    plt.yticks(fontsize=TSIZE)
    plt.ylabel("Avg-RpE", fontsize=FSIZE)
    plt.rc('legend', fontsize=FSIZE - 1)  # legend fontsize
    plt.yscale('log')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("online_adapt_sat_rem_add.pdf")
    plt.show()


def discretize_mat(mat, max_isl, t, num_sat, num_bs):
    e, _ = create_graph(mat, max_isl, num_sat, num_bs, t, print_metrics=False)
    e = np.array(e).reshape(-1, 2)
    mat = np.zeros_like(mat)
    mat[(e[:, 0], e[:, 1])] = -1
    mat[(e[:, 1], e[:, 0])] = -1
    for i in range(len(mat)):
        mat[i, i] = -np.sum(mat[i])
    return mat


def plot_regret_mip_online(runs):
    pos = generate_constellation(0.85, 0.75, 6, 11).reshape(-1, 3)
    sats_per_plane = np.repeat(11, 6)
    M = create_diag_mask(1, 1, 0, 66)
    funs_to_use = ["long_dist", "sat_dist"]
    weights = [0.6, 0.4]
    max_isl = 4
    max_dist = 0.8
    cone = False
    t=-0.5

    X_one, _, _ = laplacian_matrix_with_utilities(pos, weights, funs_to_use, max_isl, max_dist, sats_per_plane,
                                                  len(sats_per_plane), 0, cone=cone, verbose=False)

    mip_loss = pickle.load(open("./pickle_files/GUROBI_mip_loss.pickle", "rb"))
    guto_loss = []
    positions = []
    uts = []
    for i in range(runs):
        print("Current run: ", i+1)
        #mip_mat, _ = mip_topology_with_utilities(pos, weights, funs_to_use, max_isl, max_dist,
        #                            sats_per_plane, len(sats_per_plane), num_sat=66, num_bs=0, cone=cone, l_sat=1e-4)

        guto_mat, _, _ = laplacian_matrix_with_utilities(pos, weights, funs_to_use, max_isl, max_dist, sats_per_plane,
                                                  len(sats_per_plane), 0, cone=cone, verbose=False)

        P = create_connectivity_incentive(pos, 0, max_dist, cone=cone)
        cvx_ut = CvxUtility(funs_to_use, weights, 66, sats_pos=pos,
                            sats_per_plane=sats_per_plane, k=3, num_planes=len(sats_per_plane), num_bs=0)
        U = cvx_ut()
        uts.append(U)

        #mip_loss.append(loss_function(mip_mat, P, U, 1e-4*M))
        guto_loss.append(loss_function(discretize_mat(guto_mat, max_isl, t, 66, 0), P, U, 1e-4*M))
        positions.append(pos)
        pos = move_constellation(pos, 11, 6).reshape(-1, 3)

    astra_ogd_sols = admm_online_gradient_descent(runs, 50_000, X_one, uts, M, positions, 0, max_isl,
                                                  0, max_dist, 1, np.sqrt(939.3)/np.sqrt(runs), cone=cone)

    astra_ocg_sols = admm_online_conditional_gradient(runs, 50_000, X_one, uts, M, positions, 0, max_isl,
                                                  0, max_dist, 1, cone=cone)

    ogd_loss = []
    ocg_loss = []
    for i, (ogd_mat, ocg_mat) in enumerate(zip(astra_ogd_sols, astra_ocg_sols)):
        ogd_mat = discretize_mat(ogd_mat, max_isl, t, 66, 0)
        ocg_mat = discretize_mat(ocg_mat, max_isl, t, 66, 0)

        P = create_connectivity_incentive(positions[i], 0, max_dist, cone=cone)
        ogd_loss.append(loss_function(ogd_mat, P, uts[i],1e-4*M))
        ocg_loss.append(loss_function(ocg_mat, P, uts[i],1e-4*M))

    # MiP Regret
    ogd_regret = [a - m for m, a in zip(mip_loss, ogd_loss)]
    ogd_regret = np.cumsum(ogd_regret)
    ocg_regret = [a - m for m, a in zip(mip_loss, ocg_loss)]
    ocg_regret = np.cumsum(ocg_regret)

    # GUTO Regret
    ogd_g_regret = [a - m for m, a in zip(guto_loss, ogd_loss)]
    ogd_g_regret = np.cumsum(ogd_g_regret)
    ocg_g_regret = [a - m for m, a in zip(guto_loss, ocg_loss)]
    ocg_g_regret = np.cumsum(ocg_g_regret)

    purples = cm.Purples([0.6, 0.9])  # light, dark
    greens = cm.Greens([0.6, 0.9])  # light, dark

    plt.plot(np.arange(len(ogd_regret)), ogd_regret,
             label="MiP-ASTRA-OGD", color=purples[0])

    plt.plot(np.arange(len(ogd_g_regret)), ogd_g_regret,
             label="GUTO-ASTRA-OGD", color=purples[1])

    plt.plot(np.arange(len(ocg_regret)), ocg_regret,
             label="MiP-ASTRA-OCG", color=greens[0])

    plt.plot(np.arange(len(ocg_g_regret)), ocg_g_regret,
             label="GUTO-ASTRA-OCG", color=greens[1])
    plt.tight_layout()
    plt.title("Regret comparisons of ASTRA Online and GUROBI")
    plt.xticks(np.arange(0, runs+10, 10))
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(legacy='1.25')
    # plot_regret_curve(1000)
    # get_times()
    # plot_multiple_seed_sat_rem(3)
    # ablation_projected_its([5, 10, 20, 30, 40, 50, 60])
    # ablation_rho_app()
    # evaluate_on_iridium()
    # evaluate_on_starlink_data()
    # evaluate_online_algorithms_starlink()
    # plot_regret_mip_online(100)

