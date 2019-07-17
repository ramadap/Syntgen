"""
main module: generates a temporal network with known ground truth with a user specified topology

Runs for a number of steps specified by the user in user_specifications,
controlled by the methods in user_specifications
produces CSV files that can be loaded into packages like Gephi
"""

from transition import events_communities, transition_t0_t1, flow_nodes, match_communities, turnover_communities
from constants import *
from network import build_communities, build_network
from user_specifications import TIMESTEPS, DEBUG, sigma, jaccard_null_model, write_gephi, write_netgram
from output import write_nodes_csv, write_nodes_netgram, write_links_csv, printouts


def main():

    communities_t0, number_nodes_t0, nodes_t0 = build_communities()

    build_network(communities_t0, number_nodes_t0, nodes_t0)

    if write_gephi:
        write_links_csv(nodes_t0)

    for Ts.timestamp in range(1, TIMESTEPS+1):

        communities_t1, number_nodes_t1, nodes_t1 = build_communities()

        # initialize confusion matrix with deaths and births
        confusion_matrix = events_communities(communities_t0, nodes_t0, communities_t1, nodes_t1, number_nodes_t1)

        # find most similar transition
        score, global_solution = transition_t0_t1(communities_t0, communities_t1)

        # finalize confusion matrix
        flow_nodes(communities_t0, communities_t1, nodes_t1, global_solution, confusion_matrix)

        match_communities(communities_t0, communities_t1, confusion_matrix, sigma, jaccard_null_model)

        build_network(communities_t1, number_nodes_t1, nodes_t1)

        if write_gephi:
            write_links_csv(nodes_t1)

        printouts(confusion_matrix, score, communities_t0, communities_t1)

        communities_t0, nodes_t0 = turnover_communities(communities_t1, nodes_t1)

        #    [node.reset_node_links() for node in nodes_t1]

    if write_gephi:
        write_nodes_csv()

    if write_netgram:
        write_nodes_netgram()


if __name__ == "__main__":
    main()
