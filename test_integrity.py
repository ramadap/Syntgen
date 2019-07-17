from constants import *
import numpy as np


def test_changes_integrity(method):
    """ decorator method to validate the user made changes. """
    def inspect_changes(communities, nodes, dead_node_vector):
        before_nodes = nodes.copy()
        before_number_nodes = len(before_nodes)
        method(communities, nodes, dead_node_vector)
        number_deleted_nodes = before_number_nodes - len(nodes)
        assert number_deleted_nodes >= 0
        assert sum([community.get_number_nodes() for community in communities]) + number_deleted_nodes \
               == before_number_nodes
        assert len(before_nodes) == len(nodes) + number_deleted_nodes
        assert sum(dead_node_vector) == number_deleted_nodes
        return
    return inspect_changes


def test_degree_distribution(method):
    def new_degree_distribution(*args, **kwargs):
        node_degrees, node_intra_degree = method(*args, **kwargs)
        if not feasible_degrees_communities(node_degrees):
            raise Exception(" network not realizable")
        return node_degrees, node_intra_degree
    return new_degree_distribution


def feasible_degrees_deprecated(distribution):
    """
    finds if the node distribution is feasible for a cluster

    Works from lowest degree node up, computing maximum number of edges still feasible on the remaining nodes in
    the cluster.
    Can be applied to the whole network for INTER links but can result in false positives (but no false negatives)

    :param distribution: a list with node degrees
    :return: boolean feasible
    """

    distribution = distribution.copy()          # cannot change the node sequence
    total_number_edges = sum(distribution) / 2  # total number of edges required
    number_nodes = len(distribution)            # number of nodes in the community
    distribution.sort()
    if total_number_edges > number_nodes * (number_nodes - 1) / 2:
        raise Exception("Asking for more links than it is possible to create", distribution)
    link_count = 0
    feasible = True                             # assume it is feasible
    for i in range(number_nodes):
        link_count += distribution[i]           # how many links have we served?
        links_left = (number_nodes - i - 1) * (number_nodes - i - 2) / 2
        #   if DEBUG: links_left, ":", total_number_edges - link_count)
        if links_left < total_number_edges - link_count:
            print("after connecting", i, "nodes, there are not enough links to satisfy requirements",
                  (number_nodes - i - 1) * (number_nodes - i - 2) / 2, "<", total_number_edges - link_count)
            feasible = False
            break
        if link_count > total_number_edges:  # have we served everyone?
            break
    return feasible


def feasible_degrees_communities(distribution):
    """
    finds if the node distribution is feasible for a cluster

    Tests for  Erdos-Gallai condition


    :param distribution: a list with node degrees
    :return: boolean feasible
    """

    distribution = distribution.copy()          # cannot change the node sequence
    number_nodes = len(distribution)            # number of nodes in the community
    parm2 = 0
    distribution.sort(reverse=True)
    for k, edges in enumerate(distribution):
        parm2 += min(edges, k+1)
    parm1 = 0                                   # total number of edges required
    feasible = True                             # assume it is feasible
    for k in range(number_nodes):
        edges = distribution[k]
        parm1 += edges
        parm2 -= min(edges, k+1)
        if parm1 > k * (k + 1) + parm2:
            feasible = False
            break
    return feasible


def feasible_degrees_network(distribution):
    """
    finds if the node distribution is feasible for the network

    Finds the number of inter edges coming out of the graph clusters
    orders by number of inter edges per community, tests for even-number
    and if highest cluster inter degree is <= sum(remaining clusters degree)


    :param distribution: a list with node degrees
    :return: boolean feasible
    """

    distribution = distribution.copy()          # cannot change the node sequence
    distribution.sort(reverse=True)
    feasible = True                             # assume it is feasible
    sum_dist = sum(distribution)
    if sum_dist % 2:
        print("node sequence not graphical, number of inter links is odd")
        feasible = False
    if distribution[0] > sum_dist - distribution[0]:
            feasible = False
    return feasible


def is_disjoint(nodes):
    if not ALLOW_DISJOINT_COMMUNITIES:
        adj_matrix = np.zeros([len(nodes), len(nodes)])     # create an array for the adjacency matrix
        for i, node in enumerate(nodes):
            for linked_node in node.links[INTRA]:
                adj_matrix[i, nodes.index(linked_node)] = 1      # Build the adjacency matrix
        diagonal = np.sum(adj_matrix, axis=0)
        laplace = np.diag(diagonal) - adj_matrix
        ev = np.linalg.eigvals(laplace)
        null_eig = np.abs(ev) < ZERO_THRESHOLD
        components = sum(null_eig)
        if components > 1:
            print("components = ", components)
            return True
    return False
