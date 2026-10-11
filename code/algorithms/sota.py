import numpy as np
from scipy.spatial.distance import cdist


def _get_closest_satellite(sat, positions, visited_num_sats, plane_num_sats):
    pos = positions[visited_num_sats:visited_num_sats + plane_num_sats]
    cd = cdist([sat], pos)[0]
    scd = sorted(zip(np.arange(0, plane_num_sats, 1), cd), key=lambda x: x[1])
    best = scd[0]
    return best[0] + visited_num_sats


def _update_plane_info(plane_num_sats, plane_num, sats_per_plane, total_sats):
    plane_num_sats += sats_per_plane[plane_num]
    plane_num_sats = plane_num_sats % total_sats
    plane_num = (plane_num + 1) % len(sats_per_plane)
    return plane_num_sats, plane_num


def plus_grid(positions, sats_per_plane):
    prev_plane = -1
    curr_plane = 0
    next_plane = 1

    curr_total_sats = 0
    next_total_sats = sats_per_plane[curr_plane]
    prev_total_sats = np.sum(sats_per_plane[:prev_plane])
    total_sats = prev_total_sats + sats_per_plane[prev_plane]

    connected = np.zeros((len(positions), len(positions)))
    max_isl = np.zeros(len(positions))

    edge_list = []
    for i, sat in enumerate(positions):
        if i - curr_total_sats >= sats_per_plane[curr_plane]:
            curr_total_sats += sats_per_plane[curr_plane]
            curr_plane += 1

            prev_total_sats, prev_plane = _update_plane_info(prev_total_sats, prev_plane,
                                                             sats_per_plane, total_sats)
            next_total_sats, next_plane = _update_plane_info(next_total_sats, next_plane,
                                                             sats_per_plane, total_sats)

        if max_isl[i] < 4:
            cd = cdist([sat], positions[curr_total_sats:curr_total_sats + sats_per_plane[curr_plane]])[0]
            scd = sorted(zip(np.arange(0, sats_per_plane[curr_plane], 1), cd), key=lambda x: x[1])
            best = scd[1:3]
            best_neighbors = [best[0][0]+curr_total_sats, best[1][0]+curr_total_sats]

            prev_best = _get_closest_satellite(sat, positions, prev_total_sats, sats_per_plane[prev_plane])
            next_best = _get_closest_satellite(sat, positions, next_total_sats, sats_per_plane[next_plane])
            for b in [prev_best, next_best, *best_neighbors]:
                if not connected[i][b] and max_isl[b] < 4:
                    connected[i][b] = 1
                    connected[b][i] = 1
                    max_isl[b] += 1
                    max_isl[i] += 1
                    edge_list.append((i, b))

    return edge_list


def x_grid(positions, sats_per_plane):
    pprev_plane = -2
    prev_plane = -1
    curr_plane = 0
    next_plane = 1
    nnext_plane = 2

    connected = np.zeros((len(positions), len(positions)))
    max_isl = np.zeros(len(positions))

    total_sats = np.sum(sats_per_plane)
    curr_total_sats = 0
    nnext_total_sats = np.sum(sats_per_plane[:nnext_plane])
    next_total_sats = nnext_total_sats + sats_per_plane[nnext_plane]
    pprev_total_sats = np.sum(sats_per_plane[:pprev_plane])
    prev_total_sats = pprev_total_sats + sats_per_plane[pprev_plane]

    edge_list = []
    for i, sat in enumerate(positions):
        if i - curr_total_sats >= sats_per_plane[curr_plane]:
            curr_total_sats += sats_per_plane[curr_plane]
            curr_plane += 1

            next_total_sats, next_plane = _update_plane_info(next_total_sats, next_plane,
                                                             sats_per_plane, total_sats)
            prev_total_sats, prev_plane = _update_plane_info(prev_total_sats, prev_plane,
                                                             sats_per_plane, total_sats)
            nnext_total_sats, nnext_plane = _update_plane_info(nnext_total_sats, nnext_plane,
                                                               sats_per_plane, total_sats)
            pprev_total_sats, pprev_plane = _update_plane_info(pprev_total_sats, pprev_plane,
                                                               sats_per_plane, total_sats)

        if max_isl[i] < 4:
            next_best = _get_closest_satellite(sat, positions, next_total_sats, sats_per_plane[next_plane])
            nnext_best = _get_closest_satellite(sat, positions, nnext_total_sats, sats_per_plane[nnext_plane])
            prev_best = _get_closest_satellite(sat, positions, prev_total_sats, sats_per_plane[prev_plane])
            pprev_best = _get_closest_satellite(sat, positions, pprev_total_sats, sats_per_plane[pprev_plane])

            for b in [next_best, nnext_best, prev_best, pprev_best]:
                if not connected[i][b] and max_isl[b] < 4:
                    connected[i][b] = 1
                    connected[b][i] = 1
                    max_isl[b] += 1
                    max_isl[i] += 1
                    edge_list.append((i, b))

    return edge_list
