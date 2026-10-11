import inspect
import numpy as np
from operator import itemgetter
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist


def _identity(num_bodies):
    return np.diag(np.ones(num_bodies))


def _sat_dist(sats_pos):
    dist_mat = distance_matrix(sats_pos, sats_pos)
    np.fill_diagonal(dist_mat, 1)
    dist_mat = 1/dist_mat
    dist_mat = np.apply_along_axis(lambda x: x / np.max(x), 1, arr=dist_mat)
    np.fill_diagonal(dist_mat, 1e-5)
    return dist_mat


def _long_distances(sats_pos, sats_per_plane, k, num_planes, num_bs):
    sats_pos = sats_pos[num_bs:] #IF WE HAVE BASE STATIONS
    ut_mat = np.zeros((len(sats_pos), len(sats_pos)))
    curr_plane = 0
    planes_to_skip = np.arange(k, -1, -1)
    j = 0
    for i in range(len(sats_pos)):
        if (i+1) - np.sum(sats_per_plane[:curr_plane]) >= sats_per_plane[curr_plane]:
            curr_plane += 1
            j = 0

        skip_idx = j % len(planes_to_skip)
        plane_to_connect = (planes_to_skip[skip_idx] + curr_plane) % num_planes
        line_i = np.zeros(len(sats_pos))

        d = np.sum(sats_per_plane[:plane_to_connect])
        curr_sat = sats_pos[i]
        sats_to_connect = sats_pos[d:d + sats_per_plane[plane_to_connect]]
        dist = cdist([curr_sat], sats_to_connect)[0]

        if plane_to_connect == curr_plane:
            zero_dist_idx = i - d
            dist[zero_dist_idx] = -1
            dist = 1 / dist
            dist[zero_dist_idx] = 0
        else:
            dist = 1 / dist

        dist = dist / np.max(dist)
        line_i[d:d + sats_per_plane[plane_to_connect]] = dist

        ut_mat[i] = line_i
        j += 1

    np.fill_diagonal(ut_mat, 1e-5)
    #IF WE HAVE BASE STATIONS
    for _ in range(num_bs):
        ut_mat = np.insert(ut_mat, 0, 0, axis=0)
        ut_mat = np.insert(ut_mat, 1, 0, axis=1)
    return ut_mat


_STR_TO_FUN = {
    "identity": _identity,
    "sat_dist": _sat_dist,
    "long_dist": _long_distances,
}


class CvxUtility:
    def __init__(self, funs_to_use:list[str], weights, num_bodies, **kwargs):
        num_funs = len(funs_to_use)
        self.weights = np.ones(num_funs) if weights is None else weights
        self.utility = np.zeros((num_bodies, num_bodies), dtype=np.float32)
        for i in range(num_funs):
            fun = _STR_TO_FUN.get(funs_to_use[i])
            params = inspect.getfullargspec(fun)[0] # get the name of the arguments of the function
            fun_args = itemgetter(*params)(kwargs) # get all the args for the said function and call it
            if len(params) > 1:
                func_result = fun(*fun_args)
            else:
                func_result = fun(fun_args)
            self.utility += np.multiply(self.weights[i], func_result)

    def __call__(self, *args, **kwargs):
        return self.utility

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights
