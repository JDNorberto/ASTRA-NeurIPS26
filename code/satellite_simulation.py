import numpy as np


def plot_earth(ax, r=0.6371):
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    ax.plot_surface(x, y, z, alpha=0.65, color='mediumseagreen')
    ax.set_aspect('equal')


def generate_ellipsoid(a, b, n_points):
    two_pi = 2*np.pi
    step = two_pi / n_points
    X = np.arange(0, two_pi, step)
    res = []
    for x in X:
        res.append([a*np.cos(x), b*np.sin(x), 0])

    return np.array(res)


def get_3d_rot_mat(angle, axis):
    angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    return None


def generate_constellation(a, b, num_planes, sats_per_plane):
    step = 180 / num_planes
    angles = np.arange(0, 180, step)
    constellation = []
    for i, angle in enumerate(angles):
        if isinstance(sats_per_plane, list):
            points = generate_ellipsoid(a, b, sats_per_plane[i])
        else:
            points = generate_ellipsoid(a, b, sats_per_plane)
        points = np.apply_along_axis(lambda x: np.dot(x, get_3d_rot_mat(90, axis='x')), 1, points)
        points = np.apply_along_axis(lambda x: np.dot(x, get_3d_rot_mat(angle, axis='z')), 1, points)
        constellation.append(points)
    return np.array(constellation)


def move_constellation(pos, sats_pp, n_planes, step=-15):
    angles = np.arange(0, 180, 180 / n_planes)

    new_points = []
    for j in range(n_planes):
        p = pos[j * sats_pp:(j + 1) * sats_pp]
        p = np.apply_along_axis(lambda x: np.dot(x, get_3d_rot_mat(-angles[j], axis='z')), 1, p)
        p = np.apply_along_axis(lambda x: np.dot(x, get_3d_rot_mat(90, axis='x')), 1, p)
        p = np.apply_along_axis(lambda x: np.dot(x, get_3d_rot_mat(step, axis='z')), 1, p)
        p = np.apply_along_axis(lambda x: np.dot(x, get_3d_rot_mat(90, axis='x')), 1, p)
        p = np.apply_along_axis(lambda x: np.dot(x, get_3d_rot_mat(angles[j], axis='z')), 1, p)
        new_points.append(p)

    return np.array(new_points)