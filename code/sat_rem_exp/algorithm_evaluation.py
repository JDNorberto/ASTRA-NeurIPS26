import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter1d
from admm_topology_configuration import admm_online_conditional_gradient, \
    admm_online_gradient_descent
from algorithms.offline_topology_configuration import laplacian_matrix_with_utilities
from graph_analysis import create_graph
from satellite_simulation import generate_constellation, move_constellation
from utility_functions import CvxUtility


def _update_sats_per_plane(sats_per_plane, sats_rem, cs_sats_per_plane):
    for idx in sats_rem:
        smaller_planes = np.argwhere(cs_sats_per_plane < idx)
        ob_plane = smaller_planes[-1][0] + 1 if len(smaller_planes) > 0 else 0
        sats_per_plane[ob_plane] -= 1


def get_true_preds_and_positions(runs, a, b, num_sat, bs, sats_per_plane, cs_sats_per_plane, sats_rem,
                                 max_isls=4, max_dist=0.8, cone=False):
    """
    This method runs the developed method for runs "runs", considering
    a single orbital plane that is represented by an ellipse.
    :param max_dist: maximum ISL distance.
    :param num_to_rem: number of satellites to remove.
    :param seed: seed for random number generator.
    :param cone: whether to use the first or second approximation.
    :param max_isls: Maximum number of ISLs.
    :param runs: Number of runs to simulate satellite movement.
    :param a: ellipse param a.
    :param b: ellipse param b.
    :param n_satellites: number of satellites on the plane.
    :param bs: array representing position of base stations.
    :return: First X to be used by online algorithms,
             All the following predictions,
             The position vectors

    """
    sats_pos = generate_constellation(a, b, 6, 11).reshape(-1, 3) # Initial sat positions
    sats_pp = np.copy(sats_per_plane)
    funs_to_use = ["long_dist", "sat_dist"]
    new_weights = [0.6, 0.4]
    num_bs = len(bs)

    X_one, _, _ = laplacian_matrix_with_utilities(sats_pos, new_weights, funs_to_use, max_isls, max_dist,
                                                  num_sat=num_sat, num_bs=num_bs, cone=cone, verbose=False,
                                                  sats_per_plane=sats_pp, num_planes=len(sats_pp))

    true_X_values = []
    positions = []
    for i in range(runs):
        sats_pos = move_constellation(sats_pos, 11, 6).reshape(-1, 3) # Subsequent sat positions
        sats = sats_pos
        if i == 301:
            num_sat = num_sat - len(sats_rem)
            _update_sats_per_plane(sats_pp, sats_rem, cs_sats_per_plane)
        elif i == 650:
            num_sat = num_sat + len(sats_rem)
            sats_pp = sats_per_plane

        if 300 < i < 650:
            sats = np.delete(sats_pos, sats_rem, axis=0)
        #elif i >= 350:
        #    sats[sats_rem] += [0.05, 0.05, 0.05] #TODO: here we add new positions
        positions.append(sats)
        X, _, _ = laplacian_matrix_with_utilities(sats, new_weights, funs_to_use, max_isls, max_dist,
                                                  num_sat=num_sat, num_bs=num_bs, cone=cone, verbose=False,
                                                  sats_per_plane=sats_pp, num_planes=len(sats_pp))
        true_X_values.append(X)

    return X_one, true_X_values, positions


def _compare_online(X_one, mask_mat, num_sat, num_bs, max_isls, max_dist, positions, runs, true_X_values,
                    utility_mask, sats_rem, cone=False, seed=42):
    ogd_step = np.sqrt(9393)/np.sqrt(runs) #np.sqrt(9393)
    admm_sols = admm_online_gradient_descent(runs, 10_000, X_one, utility_mask, mask_mat,
                                             positions, num_bs, max_isls, 0, max_dist, 1, sats_rem,
                                             lr=ogd_step, cone=cone)
    pickle.dump(admm_sols, open(f"seed{seed}_ogd_sols_sats_rem.pickle", "wb"))
    admm_ocg_sols = admm_online_conditional_gradient(runs, 1_000_000, X_one, utility_mask, mask_mat,
                                                     positions, 0, max_isls, 0, max_dist, 1,
                                                     sats_rem, cone=cone)
    pickle.dump(admm_ocg_sols, open(f"seed{seed}_ocg_sols_sats_rem.pickle", "wb"))

    avg_m = []
    for i, mat in enumerate(admm_sols):
        if i >= 650 or i < 301:
            nsat=num_sat
        else:
            nsat = num_sat-len(sats_rem)
        edge_list, m = create_graph(mat, max_isls, nsat, num_bs, print_metrics=False)

        avg_m.append(m)
    print("[nº of edges, avg degree, density, nº cc, avg shortest path, avg clustering coefficient]")
    print("Mean: ", [round(x, 2) for x in np.mean(avg_m, axis=0)])
    print("Std: ", [round(x, 2) for x in np.std(avg_m, axis=0)])

    admm_preds = []
    admm_ocg_preds = []
    for i in range(len(true_X_values)):
        admm_norm = np.linalg.norm(true_X_values[i] - admm_sols[i])**2
        admm_ocg_norm = np.linalg.norm(true_X_values[i] - admm_ocg_sols[i])**2

        num_entries = len(np.where(true_X_values[i] != 0)[0])

        admm_preds.append(admm_norm/num_entries)
        admm_ocg_preds.append(admm_ocg_norm/num_entries)

    return admm_preds, admm_ocg_preds


def compare_online_methodology(a, b, runs, bs, num_sat, sats_per_plane, max_isls=4, max_dist=0.8,
                               cone=False, plot=True, seed=42, percent_to_rem=10):
    num_bs = len(bs)
    num_to_rem = (num_sat * percent_to_rem) // 100
    cs_sats_per_plane = np.cumsum(sats_per_plane, axis=0)
    np.random.seed(seed)
    sats_rem = np.random.choice(num_sat, num_to_rem, replace=False)

    X_one, true_X_values, positions = get_true_preds_and_positions(runs, a, b, num_sat, bs,
                                                                   sats_per_plane, cs_sats_per_plane, sats_rem,
                                                                   max_isls=max_isls, max_dist=max_dist, cone=cone)
    pickle.dump(true_X_values, open(f"seed{seed}_true_X_values_sats_rem.pickle", "wb"))

    utility_mask=[]
    sats_pp = np.copy(sats_per_plane)
    for i, p in enumerate(positions):
        if i == 301:
            _update_sats_per_plane(sats_pp, sats_rem, cs_sats_per_plane)
        elif i == 650:
            sats_pp = sats_per_plane
        cvx_ut = CvxUtility(["long_dist", "sat_dist"], [0.6, 0.4], len(p),
                            sats_per_plane=sats_pp, k=3, num_planes=len(sats_pp), num_bs=0,
                            sats_pos=p)
        utility_mask.append(cvx_ut())
    mask_mat = np.diag(np.ones(len(X_one)))
    s = time.time()
    admm_preds, admm_ocg_preds = _compare_online(X_one, mask_mat, num_sat, num_bs, max_isls, max_dist, positions,
                                                 runs, true_X_values, utility_mask, sats_rem, cone, seed)
    print("Time: ", time.time()-s)
    print(f"ADMM-OGD first value {round(admm_preds[0], 3)}, last value {round(admm_preds[-1], 3)}, "
          f"max value: {round(np.max(admm_preds), 3)}, min value: {round(np.min(admm_preds), 3)}")
    print(f"ADMM-OCG first value {round(admm_ocg_preds[0], 3)}, last value {round(admm_ocg_preds[-1], 3)}, "
          f"max value: {round(np.max(admm_ocg_preds), 3)}, min value: {round(np.min(admm_ocg_preds), 3)}")
    assert len(admm_preds) == len(admm_ocg_preds) == len(true_X_values)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(true_X_values)+1, 1), [0, *gaussian_filter1d(admm_preds, sigma=1)],
                 label='ADMM-OGD', c='purple')
        ax.plot(np.arange(0, len(true_X_values) + 1, 1), [0, *gaussian_filter1d(admm_ocg_preds, sigma=1)],
                label='ADMM-OCG', c='turquoise')
        FSIZE = 16
        TSIZE = 15
        plt.xlabel("Number of Iterations", fontsize=FSIZE)
        plt.xticks(np.arange(0, len(admm_preds)+100, 100), fontsize=TSIZE)
        plt.yticks(fontsize=TSIZE)
        plt.ylabel("Average Residual per Entry", fontsize=FSIZE)
        plt.rc('legend', fontsize=FSIZE - 1)  # legend fontsize
        plt.yscale('log')
        plt.tight_layout()
        plt.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.savefig("oco_results_single_plane.pdf")
        plt.show()


def plot_sat_removal_adapt(admm_ogd_sols, admm_ocg_sols):
    true_X_values = pickle.load(open("../true_X_values_sats_rem", "rb"))

    admm_ogd_preds = []
    admm_ocg_preds = []
    for i in range(len(true_X_values)):
        admm_norm = np.linalg.norm(true_X_values[i] - admm_ogd_sols[i])**2
        admm_ocg_norm = np.linalg.norm(true_X_values[i] - admm_ocg_sols[i])**2

        num_entries = len(np.where(true_X_values[i] != 0)[0])

        admm_ogd_preds.append(admm_norm/num_entries)
        admm_ocg_preds.append(admm_ocg_norm/num_entries)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(np.arange(0, len(true_X_values) + 1, 1),
            [0, *gaussian_filter1d(admm_ogd_preds, sigma=1)],
            label='ASTRA-OGD', c='purple')
    ax.plot(np.arange(0, len(true_X_values) + 1, 1),
            [0, *gaussian_filter1d(admm_ocg_preds, sigma=1)],
            label='ASTRA-OCG', c='darkgreen')
    FSIZE = 24
    TSIZE = 22
    plt.axvline(x=301, color='grey', linestyle='-.', linewidth=2, label='Satellite Removal')
    plt.axvline(x=650, color='grey', linestyle='--', linewidth=2, label='Satellite Addition')
    plt.xlabel("Number of Iterations", fontsize=FSIZE)
    plt.xticks(np.arange(0, len(admm_ocg_preds) + 100, 100), fontsize=TSIZE)
    plt.yticks(fontsize=TSIZE)
    plt.ylabel("Avg-RpE", fontsize=FSIZE)
    plt.rc('legend', fontsize=FSIZE - 1)  # legend fontsize
    plt.yscale('log')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1),)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    #plt.savefig("online_adapt_sat_rem_add.pdf")
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
    for f in glob.glob("./*sols_sats_rem.pickle"):
        split_f = f.split("_")

        ress = pickle.load(open(f, "rb"))
        true_X_values = pickle.load(open(split_f[0] + "_true_X_values_sats_rem.pickle", "rb"))
        ress_rpe = calculate_avg_rpe(ress, true_X_values)
        if split_f[1] == "ogd":
            ogd_rpes.append(ress_rpe)
        elif split_f[1] == "ocg":
            ocg_rpes.append(ress_rpe)

    ogd_reps = np.array(ogd_rpes).reshape(n_runs, -1)
    ocg_reps = np.array(ocg_rpes).reshape(n_runs, -1)

    ogd_reps_mean = np.array([0, *gaussian_filter1d(np.mean(ogd_reps, axis=0), sigma=.65)])
    #ogd_reps_mean = np.array([0, *np.mean(ogd_reps, axis=0)])
    ogd_reps_std = np.array([0, *gaussian_filter1d(np.std(ogd_reps, axis=0), sigma=.65)])
    #ogd_reps_std = np.array([0, *np.std(ogd_reps, axis=0)])
    ocg_reps_mean = np.array([0, *gaussian_filter1d(np.mean(ocg_reps, axis=0), sigma=.65)])
    #ocg_reps_mean = np.array([0, *np.mean(ocg_reps, axis=0)])
    ocg_reps_std = np.array([0, *gaussian_filter1d(np.std(ocg_reps, axis=0), sigma=.65)])
    #ocg_reps_std = np.array([0, *np.std(ocg_reps, axis=0)])

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
    print("""AAAAAA""")
    plt.savefig("online_adapt_sat_rem_add.pdf")
    plt.show()

def main():
    np.set_printoptions(legacy='1.25')
    seeds = np.arange(42, 45, 1)
    sats_per_plane= np.repeat(11, 6)
    num_sat = 66
    #for s in seeds:
    #    print(s)
    #    compare_online_methodology(.85, .75, 1000, [], num_sat, sats_per_plane,
    #                               max_isls=4, max_dist=0.8, cone=False, plot=False,
    #                               seed=s, percent_to_rem=10)
    plot_multiple_seed_sat_rem(len(seeds))

if __name__ == "__main__":
    main()

