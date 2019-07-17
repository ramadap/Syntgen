from constants import *
from network import Nodes, Community
from user_specifications import name_of_output_file, TIMESTEPS, print_parameters
import numpy as np


def write_links_csv(nodes: 'list[Nodes]'):
    """
    Writes links to an external file in csv format
    :param nodes:
    :return:
    """
    for node in nodes:
        for ie in [INTRA, INTER]:
            visited = []
            for target in node.links[ie]:
                if (target.node_number, node.node_number) not in visited:
                    visited.append((target.node_number, node.node_number))
                    try:
                        write_links_csv.id += 1
                    except AttributeError:
                        write_links_csv.file = open(name_of_output_file + 'links.csv', "w")
                        write_links_csv.file.write("Source;Target;Type;Id;timeset\n")
                        write_links_csv.id = 0

                    write_links_csv.file.write(str(node.node_number) + ";" + str(target.node_number) + ";Undirected;" +
                                           str(write_links_csv.id) + ";<[" + str(Ts.timestamp) + "]>\n")
    if Ts.timestamp == TIMESTEPS:
        write_links_csv.file.close()


def write_nodes_csv():

    file = open(name_of_output_file + 'nodes.csv', "w")
    file.write("id;Label;Community;timeset\n")

    for node in Nodes.list_nodes:
        timeset = "<"
        community = "<"
        comma = ""
        for snapshot in node.lifecycle:
            community += comma + "[" + str(snapshot[1]) + "," + str(snapshot[0]) + "]"
            timeset += comma + "[" + str(snapshot[1]) + "]"
            comma = ","
        community += ">"
        timeset += ">"
        file.write(str(node.node_number) + ";" + str(node.node_number) + ";"
                   + str(community) + ";" + str(timeset) + "\n")
    file.close()


def write_nodes_netgram():

    node_list_step = [[] for _ in range(TIMESTEPS+1)]  # create lists to output to matlab
    comm_list_step = [[] for _ in range(TIMESTEPS+1)]

    file = open(name_of_output_file + 'nodes.txt', "w")

    for node in Nodes.list_nodes:
        for snapshot in node.lifecycle:
            node_list_step[snapshot[1]].append(node.node_number)
            comm_list_step[snapshot[1]].append(snapshot[0])

    for node_list, comm_list in zip(node_list_step, comm_list_step):
        [file.write(str(n) + " ") for n in node_list]
        file.write("\n")
        [file.write(str(c) + " ") for c in comm_list]
        file.write("\n")
    file.close()


def print_matrix(s, columns, rows, fl):

    print()
    # Do heading
    print("     ", end="")
    for j in range(len(s[0])):
        print("%7s " % columns[j] if type(columns[j]) is str else "%7d " % columns[j], end="")
    print()
    print("     ", end="")
    for j in range(len(s[0])):
        print("--------", end="")
    print()
    # Matrix contents
    for i in range(len(s)):
        print("%3s |" % rows[i] if type(rows[i]) is str else "%3d |" % rows[i], end="")  # Row nums
        for j in range(len(s[0])):
            print(str("%"+fl) % (s[i][j]) if s[i][j] else " "*int(str(fl)[0]), end="")
        print()
        print()
    print()
    print()


def print_percentage(s, community_t0: 'list[Community]', community_t1: 'list[Community]'):
    ss = np.empty((len(s), len(s[0])), dtype='U8')
    # Do heading
    columns = []
    [columns.append(community_t1.get_community_name()) for community_t1 in community_t1]
    columns.append("D")
    rows = []
    [rows.append(community_t0.get_community_name()) for community_t0 in community_t0]
    rows.append("B")
    print("Confusion Matrix percentages")
    # compute percentages
    t0 = []
    for s_lines in s:
        t0.append(sum(s_lines))
    t = list(map(list, zip(*s)))  # transpose s
    t1 = []
    for s_columns in t:
        t1.append(sum(s_columns))
    # Matrix contents
    for i in range(len(s) - 1):
        for j in range(len(s[0]) - 1):
            ss[i][j] = " {:3.0f}/{:<3.0f}".format((100 * s[i][j] / t0[i]), 100 * s[i][j] / t1[j])
        ss[i, j+1] = " {:3.0f}/ ".format(100 * s[i][-1] / t0[i])

    for j in range(len(s[0]) - 1):
        ss[i+1, j] = "    /{:<3.0f}".format(100 * s[-1][j] / t1[j])

    print_matrix(ss, columns, rows, '8s')


def print_confusion(s, community_t0: 'list[Community]', community_t1: 'list[Community]'):
    # Do heading
    columns = []
    [columns.append(community_t1.get_community_name()) for community_t1 in community_t1]
    columns.append("D")
    rows = []
    [rows.append(community_t0.get_community_name()) for community_t0 in community_t0]
    rows.append("B")
    print("Confusion Matrix")
    print_matrix(s, columns, rows, "8d")


def print_matrices(s, community_t0, community_t1, fl):
    print()
    print("Timestep", Ts.timestamp)
    columns = []
    [columns.append(community_t1.get_community_name()) for community_t1 in community_t1]
    rows = []
    [rows.append(community_t0.get_community_name()) for community_t0 in community_t0]
    print_matrix(s, columns, rows, fl)


def print_community_events(communities):
    for community in communities:
        events = community.get_ground_truth()
        if not events:
            print("nothing recorded for community", community.get_community_name())

        else:
            for event in events:
                if not event[1]:
                    print("community", community.get_community_name(), event[0])
                elif type(event[1]) is list:
                    print("community", community.get_community_name(), event[0],
                          [ev.get_community_name() for ev in event[1]])
                else:
                    print("community", community.get_community_name(), event[0],
                          event[1].get_community_name())


def printouts(confusion_matrix, score, communities_t0, communities_t1):

    confusion_matrix_print, confusion_matrix_percentage, jaccard_index, continuity, \
        community_events_t0, community_events_t1 = print_parameters()

    if confusion_matrix_print:
        print("")
        print_confusion(confusion_matrix, communities_t0, communities_t1)
    if confusion_matrix_percentage:
        print("")
        print_percentage(confusion_matrix, communities_t0, communities_t1)
    if jaccard_index:
        print("")
        print("Jaccard Index Matrix")
        print_matrices(Community.jaccard_index[:-1, :-1], communities_t0, communities_t1, "8.5f")
    if continuity:
        print("")
        print("Continuity Matrix")
        print_matrices(Community.continuation, communities_t0, communities_t1, "8d")
    if community_events_t0:
        print("")
        print("At the end of time ", Ts.timestamp - 1)
        print_community_events(communities_t0)
    print("Variation of information:", score )
    if community_events_t1:
        print("")
        print("At the beginning of time ", Ts.timestamp)
        print_community_events(communities_t1)
