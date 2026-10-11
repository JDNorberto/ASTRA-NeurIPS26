import numpy as np
from algorithms.tc_utils import create_connectivity_incentive, create_diag_mask, loss_gradient
from utility_functions import CvxUtility


def _get_admm_matrices(total_num, num_bs, max_isl, max_bsl):
    B_upper = np.full(total_num, max_isl)
    B_upper[:num_bs] = max_bsl
    B_upper = np.diag(B_upper)

    B_lower = np.full((total_num, total_num), -1)
    np.fill_diagonal(B_lower, 0)

    Q = np.identity(total_num) - np.full((total_num, total_num), 1 / total_num)

    return Q, B_upper, B_lower


def _stopping_criteria(V, V_next, e=5e-3, verbose=False):
    V_norm = np.linalg.norm(V, "fro")
    diff_norm = np.linalg.norm(V_next - V, "fro")
    res_norm = diff_norm / V_norm if V_norm > 0 else np.inf
    if verbose:
        print(f"Stopping criteria value: {res_norm}")
    return  res_norm < e


def admm_offline(sats_pos, num_bs, rho, lambda_bs, lambda_sat, max_isl, max_bsl, max_dist, T,
                 funs_to_use, sats_per_plane, num_planes, weights=[0.75, 0.25], projected_its=20,
                 eps=5e-3, cone=False):
    """
    This method represents the ASTRA Offline algorithm.
    :param sats_pos: positions of the satellites.
    :param num_bs: number of base stations.
    :param rho: penalty parameter.
    :param lambda_bs: weight of base stations connections.
    :param lambda_sat: weight of satellites connections.
    :param max_isl: maximum number of isls.
    :param max_bsl: maximum number of bsls.
    :param max_dist: maximum isl distance.
    :param T: maximum number of iterations.
    :param funs_to_use: utility functions to use.
    :param sats_per_plane: number of satellites per plane.
    :param num_planes: number of orbital planes.
    :param weights: weights to use for the utilities
    :param projected_its: number of iterations for the projected gradient.
    :param eps: epsilon for the stopping criteria.
    :param cone: which approximation to use. False - H / True - H+C
    :return: proposed topology.
    """
    total_num = len(sats_pos)

    P = create_connectivity_incentive(sats_pos, num_bs, max_dist, cone=cone)
    cvx_ut = CvxUtility(funs_to_use, weights, total_num, sats_pos=sats_pos,
                        sats_per_plane=sats_per_plane, k=3, num_planes=num_planes, num_bs=num_bs)
    U = cvx_ut()
    M = create_diag_mask(lambda_sat, lambda_bs, num_bs, total_num)

    Q, B_upper, B_lower = _get_admm_matrices(total_num, num_bs, max_isl, max_bsl)

    U_square = np.multiply(U, U)

    X = np.zeros((total_num, total_num))
    Z = np.zeros_like(X)
    V = np.zeros_like(X)
    lr = 1 / np.sqrt(projected_its)
    for admm_t in range(T):
        #Compute the optimal value for X
        const = -np.multiply(U_square, P) - M + rho * (V - Z)
        for _ in range(projected_its):
            grad = np.multiply(U_square, X) + rho * X + const
            X -= lr * grad
            np.add(X, X.T, out=X)
            X *= 0.5
            # Project to the set of symmetric and row sum = 0 matrices (Q @ X @ Q)
            row_mean = X.mean(axis=1, keepdims=True)
            col_mean = X.mean(axis=0, keepdims=True)
            grand_mean = X.mean()

            X -= row_mean
            X -= col_mean
            X += grand_mean

        #Compute the optimal value for Z
        np.clip(X + V, B_lower, B_upper, out=Z)

        #Compute the optimal value for V
        V_next = V + X - Z

        if _stopping_criteria(V, V_next, e=eps):
            print("Stopping criteria met at iteration ", admm_t)
            _lambda = 1
            res = _lambda * X + (1 - _lambda) * Z
            return res

        V = V_next

    return X


def admm_online_gradient_descent(runs, admm_iters, X, utility_mat, mask_mat, sats_pos, num_bs, max_isl, max_bsl, max_dist,
                                 rho, lr, lambda_sat=1e-4, eps=1e-7, cone=False):
    """
    This method represents ASTRA-OGD.
    :param runs: number of runs.
    :param T: maximum number of iterations.
    :param X: first topology.
    :param utility_mat: array of utility matrices for each run.
    :param mask_mat: trace weight matrix to consider.
    :param sats_pos: positions of the satellites.
    :param num_bs: number of base stations.
    :param max_isl: maximum number of isls.
    :param max_bsl: maximum number of bsls.
    :param max_dist: maximum distance.
    :param rho: penalty parameter.
    :param lr: learning rate.
    :param cone: which approximation to use. False - H / True - H+C
    :return: array of proposed topologies with length=runs.
    """
    total_num = len(sats_pos[0])
    #lr = np.sqrt(9393) / np.sqrt(runs) - Iridium | np.sqrt(12490) / np.sqrt(runs) - Starlink

    Q, B_upper, B_lower = _get_admm_matrices(total_num, num_bs, max_isl, max_bsl)
    Z = np.zeros((total_num, total_num))
    V = np.zeros_like(Z)

    X_t = X
    sol_list = []
    for i in range(runs):
        P = create_connectivity_incentive(sats_pos[i], num_bs, max_dist, cone=cone)
        y_next = X_t - lr * loss_gradient(X_t, utility_mat[i], mask_mat, P, num_bs, bs_weight=1e0, mask_weight=lambda_sat)

        V.fill(0)
        Z.fill(0)
        # ADMM
        for t in range(admm_iters):
            W = (y_next + rho * (Z - V)) / (1 + rho)
            W = 0.5 * (W + W.T)
            # Project to the set of symmetric and row sum = 0 matrices (Q @ W @ Q)
            row_mean = W.mean(axis=1, keepdims=True)
            col_mean = W.mean(axis=0, keepdims=True)
            grand_mean = W.mean()

            W -= row_mean
            W -= col_mean
            W += grand_mean
            X_t = W
            Z = np.minimum(np.maximum(X_t + V, B_lower), B_upper)
            V_next = V + X_t - Z

            if _stopping_criteria(V, V_next, e=eps):
                print("Stopping criteria met at iteration ", t)
                break

            V = V_next

        sol_list.append(X_t)

    return sol_list


def admm_online_conditional_gradient(runs, admm_iters, X, utility_mat, mask_mat, sats_pos, num_bs, max_isls, max_bsl, max_dist,
                                     rho, lambda_sat=1e-4, eps=1e-7, cone=False):
    """
    This method represents ASTRA-OCG.
    :param runs: number of runs.
    :param T: maximum number of iterations.
    :param X: first topology.
    :param utility_mat: utility matrices for each run.
    :param mask_mat: trace weight matrix to consider.
    :param sats_pos: positions of the satellites.
    :param num_bs: number of base stations.
    :param max_isls: maximum number of isls.
    :param max_bsl: maximum number of bsls.
    :param max_dist: maximum distance.
    :param rho: penalty parameter.
    :param cone: which approximation to use. False - H / True - H+C
    :return: array of proposed topologies with length=runs.
    """
    total_num = len(sats_pos[0])
    lr = 1 / 2*(runs**(3/4))

    Q, B_upper, B_lower = _get_admm_matrices(total_num, num_bs, max_isls, max_bsl)
    Z = np.zeros((total_num, total_num))
    V = np.zeros_like(Z)

    accumulated_grad = np.zeros_like(X)

    X_t = X
    sols = []
    for i in range(runs):
        P = create_connectivity_incentive(sats_pos[i], num_bs, max_dist, cone=cone)
        accumulated_grad += loss_gradient(X_t, utility_mat[i], mask_mat, P, num_bs, bs_weight=1e0, mask_weight=lambda_sat)
        step = lr * accumulated_grad + X_t - X

        V.fill(0)
        Z.fill(0)
        # ADMM
        for t in range(admm_iters):
            W = Z - V - (step/rho)
            W = 0.5 * (W + W.T)
            # Project to the set of symmetric and row sum = 0 matrices (Q @ W @ Q)
            row_mean = W.mean(axis=1, keepdims=True)
            col_mean = W.mean(axis=0, keepdims=True)
            grand_mean = W.mean()

            W -= row_mean
            W -= col_mean
            W += grand_mean
            X_t = W
            Z = np.minimum(np.maximum(X_t + V, B_lower), B_upper)
            V_next = V + X_t - Z

            if _stopping_criteria(V, V_next, e=eps):
                print("Stopping criteria met at iteration ", t)
                break

            V = V_next

        sols.append(X_t)

    return sols

