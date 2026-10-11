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


def _insert_all(matrix, idxs_rem, val=0):
    n = matrix.shape[0]
    mask = np.ones(n + len(idxs_rem), dtype=bool)
    mask[idxs_rem] = False

    temp = np.full((n + len(idxs_rem), n + len(idxs_rem)), fill_value=val, dtype=matrix.dtype)
    temp[np.ix_(mask, mask)] = matrix

    return temp


def admm_offline(sats_pos, num_bs, rho, lambda_bs, lambda_sat, max_isl, max_bsl, max_dist, T,
                 funs_to_use, sats_per_plane, num_planes, weights=[0.75, 0.25], projected_its=20,
                 eps=5e-3, cone=False):
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
        for _ in range(projected_its):
            grad = np.multiply(U_square, X - P) - M + rho * (X - Z + V)
            X_next = X - lr * grad
            X_next = 0.5 * (X_next + X_next.T)
            #Project to the set of symmetric and row sum = 0 matrices
            X = np.dot(np.dot(Q, X_next), Q)

        #Compute the optimal value for Z
        Z = np.minimum(np.maximum(X+V, B_lower), B_upper)

        #Compute the optimal value for V
        V_next = V + X - Z

        if _stopping_criteria(V, V_next, e=eps):
            print("Stopping criteria met at iteration ", admm_t)
            return X

        V = V_next

    return X


def admm_online_gradient_descent(runs, T, X, utility_mat, mask_mat, sats_pos, num_bs, max_isl, max_bsl, max_dist,
                                 rho, sats_rem, lr, cone=False):
    total_num = len(sats_pos[0])
    #lr = np.sqrt(9393) / np.sqrt(T) #/ np.sqrt(9393)

    Q, B_upper, B_lower = _get_admm_matrices(total_num, num_bs, max_isl, max_bsl)
    Z = np.zeros((total_num, total_num))
    V = np.zeros_like(Z)

    X_t = X
    sol_list = []
    for i in range(runs):
        P = create_connectivity_incentive(sats_pos[i], num_bs, max_dist, cone=cone)
        if i == 301:
            X_t = np.delete(X_t, sats_rem, axis=0)
            X_t = np.delete(X_t, sats_rem, axis=1)

            mask_mat = np.delete(mask_mat, sats_rem, axis=0)
            mask_mat = np.delete(mask_mat, sats_rem, axis=1)

            Z = np.delete(Z, sats_rem, axis=0)
            Z = np.delete(Z, sats_rem, axis=1)

            V = np.delete(V, sats_rem, axis=0)
            V = np.delete(V, sats_rem, axis=1)

            B_lower = np.delete(B_lower, sats_rem, axis=0)
            B_lower = np.delete(B_lower, sats_rem, axis=1)

            B_upper = np.delete(B_upper, sats_rem, axis=0)
            B_upper = np.delete(B_upper, sats_rem, axis=1)

            new_total = total_num - len(sats_rem)
            Q = np.identity(new_total) - np.full((new_total, new_total), 1 / (new_total))

        if i == 650:
            X_t = _insert_all(X_t, sats_rem)

            mask_mat = _insert_all(mask_mat, sats_rem)
            np.fill_diagonal(mask_mat, 1)

            Z = _insert_all(Z, sats_rem)
            V = _insert_all(V, sats_rem)

            B_lower = _insert_all(B_lower, sats_rem, val=-1)
            np.fill_diagonal(B_lower, 0)

            B_upper = _insert_all(B_upper, sats_rem)
            np.fill_diagonal(B_upper, max_isl)

            Q = np.identity(total_num) - np.full((total_num, total_num), 1 / total_num)

        y_next = X_t - lr * loss_gradient(X_t, utility_mat[i], mask_mat, P, num_bs, bs_weight=5e1, mask_weight=1e-4)

        V.fill(0)
        Z.fill(0)
        # ADMM
        for t in range(T):
            W = (y_next + rho * (Z - V)) / (1 + rho)
            W = 0.5 * (W + W.T)
            X_t = np.dot(np.dot(Q, W), Q)
            Z = np.minimum(np.maximum(X_t + V, B_lower), B_upper)
            V_next = V + X_t - Z

            if _stopping_criteria(V, V_next, e=1e-6):
                print("Stopping criteria met at iteration ", t)
                break

            V = V_next

        sol_list.append(X_t)

    return sol_list


def admm_online_conditional_gradient(runs, T, X, utility_mat, mask_mat, sats_pos,
                                     num_bs, max_isls, max_bsl, max_dist, rho, sats_rem, cone=False):
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
        if i == 301:
            X_t = np.delete(X_t, sats_rem, axis=0)
            X = np.delete(X, sats_rem, axis=0)
            X_t = np.delete(X_t, sats_rem, axis=1)
            X = np.delete(X, sats_rem, axis=1)

            accumulated_grad = np.delete(accumulated_grad, sats_rem, axis=0)
            accumulated_grad = np.delete(accumulated_grad, sats_rem, axis=1)

            mask_mat = np.delete(mask_mat, sats_rem, axis=0)
            mask_mat = np.delete(mask_mat, sats_rem, axis=1)

            Z = np.delete(Z, sats_rem, axis=0)
            Z = np.delete(Z, sats_rem, axis=1)

            V = np.delete(V, sats_rem, axis=0)
            V = np.delete(V, sats_rem, axis=1)

            B_lower = np.delete(B_lower, sats_rem, axis=0)
            B_lower = np.delete(B_lower, sats_rem, axis=1)

            B_upper = np.delete(B_upper, sats_rem, axis=0)
            B_upper = np.delete(B_upper, sats_rem, axis=1)

            new_total = total_num - len(sats_rem)
            Q = np.identity(new_total) - np.full((new_total, new_total), 1 / (new_total))

        if i == 650:
            X_t = _insert_all(X_t, sats_rem)
            X = _insert_all(X, sats_rem)

            accumulated_grad = _insert_all(accumulated_grad, sats_rem)

            mask_mat = _insert_all(mask_mat, sats_rem)
            np.fill_diagonal(mask_mat, 1)

            Z = _insert_all(Z, sats_rem)
            V = _insert_all(V, sats_rem)

            B_lower = _insert_all(B_lower, sats_rem, val=-1)
            np.fill_diagonal(B_lower, 0)

            B_upper = _insert_all(B_upper, sats_rem)
            np.fill_diagonal(B_upper, max_isls)

            Q = np.identity(total_num) - np.full((total_num, total_num), 1 / total_num)

        accumulated_grad += loss_gradient(X_t, utility_mat[i], mask_mat, P, num_bs, bs_weight=1e0, mask_weight=1e-4)
        step = lr * accumulated_grad + X_t - X

        V.fill(0)
        Z.fill(0)
        # ADMM
        for t in range(T):
            W = Z - V - (step/rho)
            W = 0.5 * (W + W.T)
            X_t = np.dot(np.dot(Q, W), Q)
            Z = np.minimum(np.maximum(X_t + V, B_lower), B_upper)
            V_next = V + X_t - Z

            if _stopping_criteria(V, V_next, e=1e-6):
                print("Stopping criteria met at iteration ", t)
                break

            V = V_next

        sols.append(X_t)

    return sols
