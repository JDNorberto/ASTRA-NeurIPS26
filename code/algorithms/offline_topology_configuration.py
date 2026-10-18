import os
import time
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from algorithms.admm_topology_configuration import _get_admm_matrices
from algorithms.tc_utils import create_connectivity_incentive, create_diag_mask
from utility_functions import CvxUtility


def masked_trace(X, lambda_sat, lambda_bs, base_stations, total):
    """
    Base stations are always in the beginning of X.
    """
    M = lambda_sat * np.identity(total)
    return cp.trace(cp.multiply(M, X))


def utility_mat(ut_mat, num_bs):
    ut_mat[num_bs:, :num_bs] *= 1e2
    ut_mat[:num_bs, num_bs:] *= 1e2

    np.fill_diagonal(ut_mat, 1e-5)
    return ut_mat


def laplacian_matrix_with_utilities(sats_pos, weights, funs_to_use, max_isl, max_dist, sats_per_plane, num_planes,
                                    num_bs=2, lambda_sat=1e-4, lambda_bs=1e-4, eps=1e-7,
                                    cone=False, verbose=True):
    """
    This method represents GUTO. In case base stations are used,
    they should always be in the beginning of X.
    :param sats_pos: positions of the satellites.
    :param weights: weights for each utility.
    :param funs_to_use: utility functions to use.
    :param max_isl: maximum number of isls.
    :param max_dist: maximum distance.
    :param sats_per_plane: number of satellites per plane.
    :param num_planes: number of planes.
    :param num_sat: number of satellites.
    :param num_bs: number of base stations.
    :param cone: which approximation to use. False - H / True - H+C
    :param verbose: if we want to print more information.
    :return:
    """
    total_num = len(sats_pos)

    # Construct the problem.
    X = cp.Variable((total_num, total_num))

    _, B_upper, B_lower = _get_admm_matrices(total_num, num_bs, max_isl, 0)

    s = time.time()
    P = create_connectivity_incentive(sats_pos, num_bs, max_dist, cone=cone)
    prepro_time = time.time() - s

    cvx_ut = CvxUtility(funs_to_use, weights, total_num, sats_pos=sats_pos,
                        sats_per_plane=sats_per_plane, k=3, num_planes=num_planes, num_bs=num_bs)
    U = cvx_ut()

    objective = cp.Minimize(
        0.5 * cp.norm(cp.multiply(X - P, U), "fro")**2
        - masked_trace(X, lambda_sat, lambda_bs, num_bs, total_num)
    )

    # Constraints
    constraints = [X == X.T,
                   X >= B_lower,
                   X <= B_upper,
                   cp.sum(X, axis=1) == 0]
    if num_bs > 0:
        constraints.append(cp.diag(X)[:num_bs] <= 1)

    prob = cp.Problem(objective, constraints)
    _ = prob.solve(solver="SCS", verbose=False, eps=eps, eps_infeas=eps, acceleration_lookback=0)
    mat_sol = []
    if X.value is not None:
        for i in range(len(X.value)):
            row = list(map(lambda val: round(float(val), 4), X.value[i]))
            if verbose:
                a = pd.Series(row)
                print(a.value_counts())
            mat_sol.append(row)

    mat_sol = np.array(mat_sol)
    if verbose:
        print(mat_sol)
        print(np.linalg.matrix_rank(mat_sol))
    return mat_sol, sats_pos, [prob.solver_stats.num_iters, prob.solver_stats.solve_time, prepro_time]


def mip_topology_with_utilities(sats_pos, new_weights, funs_to_use, max_isl, max_dist, sats_per_plane, num_planes,
                                    num_sat=98, num_bs=2, cone=False, l_sat=1e-4):
    """
    This method represents the GUROBI solution. In case base stations are used,
    they should always be in the beginning of X.
    :param sats_pos: positions of the satellites.
    :param new_weights: weights for each utility.
    :param funs_to_use: utility functions to use.
    :param max_isl: maximum number of isls.
    :param max_dist: maximum distance.
    :param sats_per_plane: number of satellites per plane.
    :param num_planes: number os planes.
    :param num_sat: number of satellites.
    :param num_bs: number of base stations.
    :param cone: which approximation to use. False - H / True - H+C.
    :param l_sat: value of the matrix Lambda to consider.
    :return:
    """
    os.environ["GRB_LICENSE_FILE"]= "" #Add your gurobi license here
    total_num = num_sat + num_bs

    model = gp.Model("topology_optimization")
    X = model.addMVar((total_num, total_num), vtype=GRB.INTEGER)

    _, B_upper, B_lower = _get_admm_matrices(total_num, num_bs, max_isl, 0)

    P = create_connectivity_incentive(sats_pos, num_bs, max_dist, cone=cone)

    cvx_ut = CvxUtility(funs_to_use, new_weights, total_num, sats_pos=sats_pos,
                        sats_per_plane=sats_per_plane, k=3, num_planes=num_planes, num_bs=num_bs)
    U = cvx_ut()

    expr = U * (X - P)
    first_term = 0.5 * (expr * expr).sum()
    M = create_diag_mask(l_sat, 1, 0, total_num)
    second_term = gp.quicksum(M[i, i] * X[i, i] for i in range(M.shape[0]))
    model.setObjective(first_term-second_term, GRB.MINIMIZE)
    model.addConstr(X == X.T)
    for i in range(total_num):
        for j in range(total_num):
            if i == j:
                X[i, j].LB = 0
                X[i, j].UB = 4
            else:
                X[i, j].LB = -1
                X[i, j].UB = 0
    model.addConstr(X.sum(axis=1) == 0)

    model.Params.OutputFlag = 0
    model.Params.MIPGapAbs = 1e-7
    model.Params.FeasibilityTol = 1e-7
    model.Params.OptimalityTol = 1e-7
    model.Params.BarConvTol = 1e-7

    model.optimize()
    x_sol = X.X
    edge_list = []
    for i in range(len(x_sol)):
        connections = np.where(x_sol[i] == -1)[0]
        if len(connections) > 0:
            edge_list.extend([(i, c) for c in connections])

    return x_sol, edge_list
