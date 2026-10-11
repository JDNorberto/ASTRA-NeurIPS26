import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.spatial.distance
from satellite_simulation import plot_earth


class Hyperplane:
    """
    Represents a hyperplane of form: a*x + b*y + c*z + d = 0
    """

    def __init__(self, x_slope, y_slope, z_slope, bias):
        self.x_slope = x_slope
        self.y_slope = y_slope
        self.z_slope = z_slope
        self.bias = bias

    def evaluate(self, points):
        if self.z_slope is None:
            result = np.dot(points, [self.x_slope, self.y_slope]) + self.bias
        else:
            result = np.dot(points, [self.x_slope, self.y_slope, self.z_slope]) + self.bias
        negative = np.where(result < 0)
        return negative


def bs_preprocessing(X, bs_idx, plot=False):
    x, y, z = X[bs_idx]
    coeffs = np.array([x, y, z])
    hyperplane = Hyperplane(*coeffs, - coeffs[0] ** 2 - coeffs[1] ** 2 - coeffs[2] ** 2)
    negatives = hyperplane.evaluate(X)[0]
    if plot:
        ax = plt.axes(projection='3d')
        positive = np.delete(X, [*negatives, bs_idx], axis=0)
        ax.scatter(positive[:, 0], positive[:, 1], positive[:, 2], c='blue')
        ax.scatter(x, y, z, c='black')
        ax.scatter(X[negatives, 0], X[negatives, 1], X[negatives, 2], c='orange')
        plot_earth(ax, r=0.5)
        plt.show()
    res = []
    for x in negatives:
        res.extend([(bs_idx, x), (x, bs_idx)])
    return res


def bs_3d_preprocessing(X, num_bs, radius, max_dist, cone=False, plot=False):
    zero_entries = []
    for i in range(len(X)):
        if i < num_bs:
            bs_zero_entries = bs_preprocessing(X, i, plot=plot)
            zero_entries.extend(bs_zero_entries)
            continue
        negatives = np.zeros(len(X))
        x, y, z = X[i]
        normal = np.array([x, y, z])
        coeffs = radius * (normal / np.linalg.norm(normal))
        hyperplane = Hyperplane(*coeffs, - coeffs[0] ** 2 - coeffs[1] ** 2 - coeffs[2] ** 2)
        negatives[hyperplane.evaluate(X)[0]] += 1

        if cone:
            sat_height = np.linalg.norm(normal)
            cone_angle = math.asin(radius / sat_height)
            cos_cone_angle = math.cos(cone_angle)

            sat_centered_X = X - normal
            sat_centered_X = np.apply_along_axis(lambda a:  a / np.linalg.norm(a) if np.linalg.norm(a) != 0 else a,
                                                 arr=sat_centered_X, axis=1)
            result = np.dot(sat_centered_X, -normal / np.linalg.norm(-normal))
            negatives[np.where(result > cos_cone_angle)[0]] += 1

        num_neg = 2 if cone else 1

        negative = np.where(negatives == num_neg)[0]
        for n in negative:
            if n >= num_bs:
                zero_entries.extend([(i, n), (n, i)])

        positive = np.delete(np.arange(0, X.shape[0], 1), [*negative, i], axis=0)
        dist = scipy.spatial.distance.cdist([normal], X[positive])
        dist_negatives = []
        for n, d in zip(positive, dist[0]):
            if d >= max_dist:
                dist_negatives.extend([(i, n), (n, i)])
        zero_entries.extend(dist_negatives)

        if plot:
            ax = plt.axes(projection='3d')
            if len(dist_negatives) > 0:
                dn = np.array(dist_negatives)
                negative = np.append(negative, dn[:, 1])
            positive = np.delete(X, [*negative, i], axis=0)
            ax.scatter(positive[:, 0], positive[:, 1], positive[:, 2], c='blue')
            ax.scatter(x, y, z, c='black')
            ax.scatter(X[negative, 0], X[negative, 1], X[negative, 2], c='orange')
            plot_earth(ax, r=0.5)
            plt.show()
    return zero_entries


def create_connectivity_incentive(sats_pos, num_bs, max_dist, cone=False):
    sats_pos = np.array(sats_pos)
    zero_entries = np.array(bs_3d_preprocessing(sats_pos, num_bs, radius=0.6371, max_dist=max_dist, cone=cone, plot=False))
    zero_idxs = (zero_entries[:, 0], zero_entries[:, 1])
    P = -np.ones(shape=(len(sats_pos), len(sats_pos)))
    P[zero_idxs] = 0
    np.fill_diagonal(P, 0)
    return P


def create_diag_mask(lambda_sat, lambda_bs, base_stations, total):
    lambda_list = np.repeat(lambda_sat, total)
    lambda_list[:base_stations] = lambda_bs
    return np.diag(lambda_list)


def loss_gradient(X, U, M, P, num_bs, bs_weight, mask_weight):
    r"""
    Gradient of the loss:

    .. math::
        0.5\left\| (X - P) \odot U \right\|_F^2 - \operatorname{Tr}(\Lambda \odot X)
    :param X: The generated Laplacian matrix.
    :param U: The utility matrix.
    :param M: The mask matrix.
    :return: The gradient of the loss.
    """
    U_square = np.multiply(U, U)

    #Note: It does not matter if weights are multiplied by U or U_square.
    #      If they are multiplied in U, do not forget they are then squared.
    if num_bs > 0:
        U_square[num_bs:, :num_bs] *= bs_weight
        U_square[:num_bs, num_bs:] *= bs_weight

    return np.multiply(X - P, U_square) - mask_weight*M