import io
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image
from satellite_simulation import plot_earth


def graph_statistics(G: nx.Graph, print_metrics=False):
    """
    Calculate graph statistics.
    :param G: Graph.
    :param print_metrics: If you want to print some graph metrics.
    :return: array with the following statistics:
        [number of edges, avg degree, density, number of connected components, avg shortest path,
        avg clustering coefficient]
    """
    deg = np.array(G.degree())
    n_cc = nx.number_connected_components(G)
    if n_cc == 1: avg_sp = nx.average_shortest_path_length(G)
    else: avg_sp = 0

    metrics = [ G.number_of_edges(), sum(deg[:, 1]) / len(deg), nx.density(G),
             n_cc, avg_sp, nx.average_clustering(G) ]

    if print_metrics:
        print("---------- Graph Statistics ----------")
        print("Number of Edges: ", metrics[0])
        print("Avg Node Degree: ", round(metrics[1], 3))
        print("Graph Density: ", round(metrics[2], 3))
        print("Number of connected components: ", metrics[3])
        print("Average shortest Path Length: ", round(metrics[4], 3))
        print("Average Clustering: ", round(metrics[5], 3))
        print("--------------------------------------")

    return metrics


def plot_graph(edge_list, positions, num_bs, show_plot=True):
    """
    This method plots the graph represented by edge_list and positions.
    :param show_plot: If the plot should be shown.
    :param edge_list: The list of edges.
    :param positions: The position of the nodes
    :param num_bs: The number of base stations
    """
    ax = plt.axes(projection='3d')
    plot_earth(ax)
    for i, j in edge_list:
        x, y, z = positions[i]
        tx, ty, tz = positions[j]
        if i < num_bs or j < num_bs:
            ax.plot([x, tx], [y, ty], [z, tz], marker='o', linestyle='-', color='black')
        else:
            ax.plot([x, tx], [y, ty], [z, tz], 'ro-')
    sat_line = mlines.Line2D([], [], color='red', linestyle='-', label='Sat-Sat Connection')
    sat_dot = mlines.Line2D([], [], color='black', marker='o', markersize=5,linestyle='None', label='Satellite')
    bs_line = mlines.Line2D([], [], color='black', linestyle='-', label='Bs-Sat Connection')
    pos = np.array(positions)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="black")
    if num_bs > 0:
        plt.legend(handles=[sat_line, bs_line, sat_dot], loc='upper right')
    else:
        plt.legend(handles=[sat_line, sat_dot], loc='upper right')
    ax.set_aspect('equal')
    if show_plot:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True)
        plt.savefig(f"collapsed_topology.pdf")
        plt.show()


def animate_graph(solutions, positions, max_isls, num_sat, num_bs, gif_name="topology_gif"):
    frames = []
    for i, solution in enumerate(solutions):
        edge_list, _ = create_graph(solution, max_isls, num_sat, num_bs, print_metrics=False)
        plot_graph(edge_list, positions[i], num_bs, show_plot=False)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        frames.append(Image.open(buf))

    frames[0].save(f"{gif_name}.gif", format='GIF', append_images=frames, save_all=True, duration=200, loop=0)


def create_graph(mat, max_isls=4, num_sat=18, num_bs=2, threshold=-0.5, print_metrics=True):
    """
    This method creates a graph from a Laplacian matrix. The connections
    are chosen greedily, based on the flattened matrix.
    :param threshold: Threshold to define valid ISLs.
    :param mat: The Laplacian matrix.
    :param max_isls: Maximum number of ISLs to consider.
    :param num_sat: Number of satellites.
    :param num_bs: Number of base stations.
    :param print_metrics: If you want to print some graph metrics.
    :return: Edge list and graph metrics
    """
    visited = np.zeros_like(mat, dtype=np.int32)
    connections = np.zeros(num_sat + num_bs, dtype=np.int32)
    edge_list = []

    indices = list(np.ndindex(mat.shape))
    # flat_mat[..] = (value, i, j)
    flat_mat = list(zip(mat.flatten(), indices))
    sorted_flat_mat = sorted(flat_mat, key=lambda x: x[0])
    sorted_flat_mat = list(filter(lambda x : x[0] <= threshold, sorted_flat_mat))
    for _, idxs in sorted_flat_mat: #can be more efficient
        first_idx, second_idx = idxs
        if not visited[first_idx][second_idx]:
            if connections[first_idx] >= max_isls or connections[second_idx] >= max_isls:
                continue
            else:
                edge_list.append((first_idx, second_idx))
                visited[first_idx][second_idx] = 1
                visited[second_idx][first_idx] = 1
                if first_idx < num_bs:
                    connections[first_idx] = max_isls #bs only connects with one sat
                elif second_idx < num_bs:
                    connections[second_idx] = max_isls

                connections[first_idx] += 1
                connections[second_idx] += 1
    G = nx.Graph(edge_list)
    G.remove_edges_from(nx.selfloop_edges(G))

    metrics = graph_statistics(G, print_metrics=print_metrics)
    return edge_list, metrics

