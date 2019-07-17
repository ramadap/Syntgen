"""
This module aggregates all network related objects, methods and functions.

It includes communities, nodes, and all methods to build a network with nodes and links

"""
from random import sample
import numpy as np
from bisect import bisect, bisect_left
from user_specifications import node_distribution, DEBUG
from test_integrity import feasible_degrees_communities, feasible_degrees_network, is_disjoint
from constants import *
from user_specifications import community_distribution, degree_affinity, assortativity, stop_if_disjoint, \
    retry_new_degree_distribution


class Nodes:
    """

    """
    highest_node_number = 1  # to serialize nodes
    total_intra = 0  # sum of intra degrees
    total_degree = 0  # sum of total degrees
    list_nodes = []  # list of all nodes ever created
    alfa_assortativity = 1  # alfa_assortativity distribution parameters for assortativity
    beta_assortativity = 1  # beta_assortativity distribution ""
    alfa_affinity = 1  # alfa_assortativity distribution parameters for t to t+1 degree affinity
    beta_affinity = 1  # beta_assortativity distribution ""

    def __init__(self, community):
        self.number_links = [0, 0]  # [intra, inter]
        self.number_links_work = [0, 0]  # [intra, inter]
        self.links = [[], []]  # [intra, inter]
        self.community = community
        self.node_number = Nodes.highest_node_number
        self.timestamp = Ts.timestamp
        Nodes.highest_node_number += 1
        self.alive = True
        self.lifecycle = []
        Nodes.list_nodes.append(self)

    def get_community(self):
        return self.community

    def change_community(self, community):
        self.community = community

    def reset_node_links(self):
        self.number_links_work = [0, 0]  # [intra, inter]
        self.links = [[], []]

    def set_lifecycle(self):
        self.lifecycle.append((self.community.community_name, Ts.timestamp))

    def kill_node(self):
        self.alive = False  # mark it dead (to be removed after output)
        self.community.nodes.remove(self)  # and remove it from the community t0 node list

    def kill_node(self, nodes: 'list[Nodes]'):
        """
        removes a dead node from the community nodes list and the nodes list.

        If it is the last node in the community deletes the community
        :param node: node to be deleted
        :param nodes: list of nodes from where to delete it
        :return:
        """
        self.community.nodes.remove(self)
        if self.community.get_number_nodes() == 0:
            del self.community
        nodes.remove(self)


class Community:
    highest_community_number = 0  # controls community numbers
    top_community_name = 1  # community name for time series
    jaccard_index = None  # jaccard matrix for a T to T+1 transition
    continuation = None  # same for continuation matrix

    def __init__(self, nodes):
        self.nodes = []
        self.timestamp = (Ts.timestamp, Ts.timestamp)  # community life span
        self.total_nodes = nodes  # required nodes
        self.community_number = Community.highest_community_number
        self.alive = True
        self.death_by = ''
        Community.highest_community_number += 1
        self.community_events = []
        self.community_name = Community.top_community_name  # initial name before lifecycle
        Community.top_community_name += 1

    def get_number_nodes(self):
        return len(self.nodes)

    def get_number_missing_nodes(self):
        return self.total_nodes - self.get_number_nodes()

    def set_number_total_nodes(self, number_nodes):
        self.total_nodes = number_nodes

    def set_community_name(self, *community_name):
        if community_name:
            self.community_name = community_name[0]
        else:
            self.community_name = Community.top_community_name
            Community.top_community_name += 1

    def get_community_number(self):
        return self.community_number

    def community_death(self, reason):
        self.alive = False
        self.timestamp = (self.timestamp[0], Ts.timestamp)
        self.death_by = reason

    def get_number_links(self, ie=2):
        if ie == 2:
            return sum(self.get_number_links(INTRA) + self.get_number_links(INTER))
        else:
            return sum([node.number_links[ie] for node in self.nodes])

    def set_ground_truth(self, event: 'str', communities: 'list[Community]'):
        if event == 'reset':
            self.community_events = []
        else:
            self.community_events.append((event, communities))

    def get_ground_truth(self) -> 'list[tuple]':
        if len(self.community_events) > 0:
            return self.community_events
        else:
            return []

    def get_community_name(self):
        return self.community_name


def build_communities() -> ('list[Community]', 'int', 'list[Nodes]'):
    """
    Creates the list of communities according to user size distribution
    :return: list of communities
    """
    communities = []
    community_sizes = community_distribution()
    community_sizes.sort()
    number_nodes = sum(community_sizes)
    if DEBUG:
        print("Number of communities: ", len(community_sizes), ", Number of nodes: ", number_nodes)
    # Build Community and node lists that cross reference each other
    Community.highest_community_number = 0
    for size in community_sizes:
        communities.append(Community(size))
        if DEBUG:
            print("Community with", size, " nodes")

    return communities, number_nodes, []


def create_new_nodes(community, nodes, number_nodes):
    # Build Community and node lists that cross reference each other
    assert community.total_nodes >= community.get_number_nodes() + number_nodes
    for i in range(number_nodes):
        nodes += [Nodes(community)]
        community.nodes.append(nodes[-1])


def degree_assignment(number_nodes_in_community: 'int', previous_intra_degrees: 'list[int]',
                      previous_inter_degrees: 'list[int]', candidate_intra_degrees: 'list[int]',
                      candidate_inter_degrees: 'list[int]', degree_affinity: 'double') -> 'list[int],list[int]':
    """
    Selects intra and inter degrees sequences for nodes in a community. Does it randomly within candidate degrees
    except if degree_affinity is True in which case popular nodes tend to remain popular.

    Receives list of candidate degrees and previous node degrees. New nodes have a previous degree of zero.
    Must return an in-order matching list of new degrees
    :param number_nodes_in_community: Total number of nodes to assign degrees
    :param previous_intra_degrees:
    :param previous_inter_degrees:
    :param candidate_intra_degrees:
    :param candidate_inter_degrees:
    :param degree_affinity: if node degree affinity across intervals should be attempted
    :return: lists of intra and inter degrees
    """

    intra_degrees = []
    inter_degrees = []
    # ensure all is kosher
    assert (number_nodes_in_community == len(previous_intra_degrees) == len(previous_inter_degrees))

    # get one  pair (intra/inter) randomly from candidates
    # generate a random list to avoid biasing for other communities
    for _ in range(number_nodes_in_community):
        k = np.random.randint(len(candidate_intra_degrees))
        intra_degrees.append(candidate_intra_degrees[k])
        del candidate_intra_degrees[k]
        inter_degrees.append(candidate_inter_degrees[k])
        del candidate_inter_degrees[k]

    # uses random list to sample according to beta distribution for t to t+1 degree correlation
    if degree_affinity:
        consistent_intra_degrees = [None]*number_nodes_in_community
        consistent_inter_degrees = [None]*number_nodes_in_community
        degrees = [intra_degree + inter_degree
                   for intra_degree, inter_degree in zip(previous_intra_degrees, previous_inter_degrees)]
        node_indexes_previous = sorted(range(len(degrees)), key=degrees.__getitem__, reverse=True)
        degrees = [intra_degree + inter_degree
                   for intra_degree, inter_degree in zip(intra_degrees, inter_degrees)]
        node_indexes_now = sorted(range(len(degrees)), key=degrees.__getitem__, reverse=True)
        for i in range(number_nodes_in_community):
            # sample beta_assortativity according to user parameters
            selected_index = np.random.beta(Nodes.alfa_affinity, Nodes.beta_affinity)
            selected_index *= len(node_indexes_now) - 1  # scale to candidate nodes size
            selected_index = (int(round(selected_index)))
            consistent_intra_degrees[node_indexes_previous[i]] = intra_degrees[node_indexes_now[selected_index]]
            consistent_inter_degrees[node_indexes_previous[i]] = inter_degrees[node_indexes_now[selected_index]]
            del node_indexes_now[selected_index]

        return consistent_intra_degrees, consistent_inter_degrees

    else:
        return intra_degrees, inter_degrees


def build_nodes(communities: 'list[Community]', number_nodes: 'int', nodes: 'list[Nodes]'):
    """
    Assigns node intra and inter degree distribution to community nodes

    Creates the node objects and list of nodes per community.
    Updates the the number of intra and inter links for each node. Does not link them yet. Just the numbers.
    :param communities:
    :param number_nodes:
    :param nodes:
    :return:
    """

    # fill the communities with nodes , if new nodes are needed
    for community in communities:
        create_new_nodes(community, nodes, community.get_number_missing_nodes())

    # try to assign node degrees, if network not graphable
    # request new distributions from user "retry_new_degree_distribution" times
    for retries in range(retry_new_degree_distribution):
        succeed = True  # flag of successful network creation, assume True
        # Create distribution of intra and inter node degrees (node_degrees, node_intra_degrees)
        # create a list of node degrees (total, intra-community) to assign to the network nodes
        node_degrees, node_intra_degrees = node_distribution([community.total_nodes for community in communities],
                                                             retries)

        Nodes.total_degree = sum(node_degrees)
        Nodes.total_intra = sum(node_intra_degrees)
        # communities sorted by size, sort now nodes by intradegree
        node_indexes = np.argsort(node_intra_degrees)
        node_inter_degrees = ([node_degrees[i] - node_intra_degrees[i] for i in node_indexes])
        node_intra_degrees = [node_intra_degrees[i] for i in node_indexes]
        previous_degree_sequence = []
        current_degree_sequence = []

        # Assign degree to nodes
        if ADJUSTED_MAX_DEGREE:
            max_community_size = max([community.get_number_nodes() for community in communities])
            max_node_degree = max(node_intra_degrees)
            ratio = max_node_degree / max_community_size
            if ratio > 1:
                raise Exception("maximum node intra degree invalid (> max community size)")
            print("ratio com/degree size", ratio)
        else:
            ratio = 1
        for community in communities:
            number_nodes_in_community = community.get_number_nodes()  # number of nodes in community
            assert community.total_nodes == number_nodes_in_community
            # section the list of the node distribution so that
            # we do not request an intra degree > community cardinality
            # index of last candidate
            candidate_index = bisect(node_intra_degrees, (number_nodes_in_community - 1) * ratio)
            if candidate_index < number_nodes_in_community:
                if DEBUG:
                    print(candidate_index, number_nodes_in_community)
                raise Exception("network not realizable: not enough nodes with intra degrees to fill community ",
                                community.community_number,
                                "relax community and degree distribution to feasible levels")

            candidate_intra_degrees = node_intra_degrees[:candidate_index]  # candidate degrees for assignment
            candidate_inter_degrees = node_inter_degrees[:candidate_index]

            node_intra_degrees = node_intra_degrees[candidate_index:]  # remove from overall candidate list
            node_inter_degrees = node_inter_degrees[candidate_index:]

            # get previous node degrees
            previous_intra_degrees = [node.number_links[INTRA] for node in community.nodes]
            previous_inter_degrees = [node.number_links[INTER] for node in community.nodes]

            # get new assignment
            # intra/inter degrees is the ordered list to assign to the community nodes
            intra_degrees, inter_degrees = degree_assignment(number_nodes_in_community, previous_intra_degrees,
                                                             previous_inter_degrees, candidate_intra_degrees,
                                                             candidate_inter_degrees, degree_affinity)

            node_intra_degrees = candidate_intra_degrees + node_intra_degrees  # add unused degrees
            node_inter_degrees = candidate_inter_degrees + node_inter_degrees
            if Ts.timestamp > 0:
                current_degree_sequence += [i + o for i, o in zip(intra_degrees, inter_degrees)]
                previous_degree_sequence += [i + o for i, o in zip(previous_intra_degrees, previous_inter_degrees)]

            for i, node in enumerate(community.nodes):  # assign node degrees
                node.number_links[INTRA] = intra_degrees[i]
                node.number_links[INTER] = inter_degrees[i]

            # test for an even number of intra links and adjust for Configuration model to work
            while community.get_number_links(INTRA) % 2 != 0:
                try_node = np.random.randint(number_nodes_in_community)
                if np.random.randint(0, 2) == 1:
                    if community.nodes[try_node].number_links[INTRA] < number_nodes_in_community - 1:
                        community.nodes[try_node].number_links[INTRA] += 1
                        Nodes.total_intra += 1
                        Nodes.total_degree += 1
                        if DEBUG:
                            print("node ", community.nodes[try_node].node_number, " intralink adjusted up")
                else:
                    if community.nodes[try_node].number_links[INTRA] > 0:
                        community.nodes[try_node].number_links[INTRA] -= 1
                        Nodes.total_intra -= 1
                        Nodes.total_degree -= 1
                        if DEBUG:
                            print("node ", community.nodes[try_node].node_number, " intralink adjusted down")
            degree_distribution = [nodes.number_links[INTRA] for nodes in community.nodes]
            if not feasible_degrees_communities(degree_distribution):
                print("community", community.community_name, " degree distribution not realizable on try #", retries+1,
                                                             " must relax parameters:", degree_distribution)
                succeed = False
                break
        if not succeed:
            continue
        # here adjust total number of interlinks to even
        while sum([community.get_number_links(INTER) for community in communities]) % 2 != 0:
            try_node = np.random.randint(number_nodes)
            if np.random.randint(0, 2) == 1:
                if nodes[try_node].number_links[INTER] < number_nodes:
                    nodes[try_node].number_links[INTER] += 1
                    Nodes.total_degree += 1

                    if DEBUG:
                        print("node ", nodes[try_node].node_number, " interlink adjusted up")
            else:
                if nodes[try_node].number_links[INTER] > 0:
                    nodes[try_node].number_links[INTER] -= 1
                    Nodes.total_degree -= 1
                    if DEBUG:
                        print("node ", nodes[try_node].node_number, " interlink adjusted down")

        degree_distribution = [sum([node.number_links[INTER] for node in community.nodes]) for community in communities]
        if not feasible_degrees_network(degree_distribution):
            print("Network degree distribution not realizable on try #", retries+1, " must relax parameters:",
                  degree_distribution)
            succeed = False
            continue
        else:
            print("Network creation succeeded after", retries+1,  "tries")
            break

    if not succeed:
        raise Exception("Network creation failed, non-graphable, after ", retry_new_degree_distribution, "tries")

    if Ts.timestamp > 0:
        pearsons_coef = np.corrcoef(previous_degree_sequence, current_degree_sequence)[0][1]
        print("Pearson's node degree corrrelation at time ", Ts.timestamp - 1, "to", Ts.timestamp, pearsons_coef)
    if DEBUG:
        print("build nodes completed")


def generate_links_intra(communities: 'list[Community]', nodes: 'list[Nodes]'):
    """
    Generates the intra links per community according to degree specification.

    :param communities:
    :param nodes:
    :return:
    """
    if DEBUG:
        print("entering intra links generation")

    complete = False
    cycle = 0
    while not complete:
        complete = True
        cycle += 1
        cycle_per_community = 0
        for community in communities:
            if cycle_per_community == MAX_CYCLES:
                raise Exception('Community', community.community_number, ' not linked after ', cycle, 'cycles')
            assert community.total_nodes == community.get_number_nodes()
            if DEBUG:
                print("intra nodes for community", community.community_number, " with ", community.total_nodes, "nodes")
            community.nodes.sort(key=lambda x: x.number_links[INTRA] - x.number_links_work[INTRA], reverse=True)
            to_nodes = community.nodes
            for from_node in community.nodes:
                # only invokes Conf model if links are required
                if from_node.number_links[INTRA] > from_node.number_links_work[INTRA]:
                    # if DEBUG: print(" assigning INTRA links for node ", from_node.node_number)
                    # print(from_node, to_nodes)
                    if not configuration_model(from_node, to_nodes, INTRA):
                        complete = False
                        if DEBUG:
                            print("incomplete intra link assignments for community ", community.community_number,
                                  ", will try to fix later")
            if complete:  # test if link assignment complete
                if is_disjoint(community.nodes):
                    if stop_if_disjoint:
                        raise Exception("disjoint community, must relax parameters")
                    else:
                        print("warning: disjoint community", community.community_name, ", increase average degree")
            if DEBUG:
                print("Cycle:", cycle, ", Community", community.get_community_number(), "required links: ",
                      sum([node.number_links[INTRA] for node in community.nodes]),
                      ", done ", sum([len(node.links[INTRA]) for node in community.nodes]))
    if DEBUG:
        print("intra links generation completed", Nodes.total_intra, sum([len(node.links[INTRA]) for node in nodes]),
              "in", cycle, "cycles")


def generate_links_inter(nodes: 'list[Nodes]'):
    """
     Generates the inter links for the whole network according to degree specification.
    :param nodes:
    :return:
    """
    if DEBUG:
        print("entering inter links generation")
    nodes.sort(key=lambda x: x.number_links[INTER], reverse=True)
    not_complete = True
    cycle = 0
    while not_complete:
        if cycle == MAX_CYCLES:
            raise Exception('Network not linked after ', cycle, 'cycles')
        old_community = []
        not_complete = False
        incomplete_nodes = [node for node in nodes if node.number_links[INTER] > node.number_links_work[INTER]]
        incomplete_nodes.sort(key=lambda x: x.number_links[INTER] - x.number_links_work[INTER], reverse=True)
        count = len(incomplete_nodes)
        if DEBUG:
            print("Linking remaining", count, "nodes, with ", sum([node.number_links[INTER] -
                                                                   node.number_links_work[INTER]
                                                                   for node in nodes]), "links")
        for from_node in incomplete_nodes:
            if from_node.number_links[INTER] > from_node.number_links_work[INTER]:  # need to re-check
                if from_node.get_community() is not old_community:
                    old_community = from_node.get_community()
                    to_nodes = []
                    for candidate_node in nodes:
                        if from_node.get_community() is not candidate_node.get_community():
                            to_nodes.append(candidate_node)
                if len(to_nodes) > 0:
                    if not configuration_model(from_node, to_nodes, INTER):
                        not_complete = True
                        if DEBUG:
                            print("inter link assignment for node ", from_node.node_number,
                                  "required re-wiring, will try to fix donor node later")
                else:
                    raise Exception("network Inter connections not realizable")
        cycle += 1
        if DEBUG:
            print("Cycle", cycle, "Required links: ",
                  Nodes.total_degree - Nodes.total_intra, ", got",
                  sum([len(node.links[INTER]) for node in nodes]))

    if DEBUG:
        print("inter links generation completed", Nodes.total_degree - Nodes.total_intra,
              sum([len(node.links[INTER]) for node in nodes]), "in ", cycle, "cycles")


def configuration_model(from_node: 'Nodes', to_nodes: 'list[Nodes]', ie: 'int') -> 'bool':
    """
    Matches stubs to link nodes. Different approach if assortativity is requested.

    This is a modified configuration model in that we do not allow self loops or multi-edges
    :param from_node: node to link
    :param to_nodes: list of candidate nodes
    :param ie: INTRA or INTER
    :return: True if successful, False if it had to rewire
    """

    succeed = True
    # ie=0 intra nodes ie=1 inter nodes
    # number of links required
    from_number_links = from_node.number_links[ie] - from_node.number_links_work[ie]
    if DEBUG:
        print("entering configuration model for node ", from_node.node_number, " phase ", ie, " with ",
              from_number_links, "links to resolve")
    edge_creation_complete = True           # assume links will be fulfilled

    # build stubs "proxies" as a summation list, for easy scanning: Entry(n+1) = sum(Entry(n)) n=[0: n]

    to_nodes_stripped = []
    # order node candidate list by degree similarity and exclude selfloops and multi-edges
    for i, to_node in enumerate(to_nodes):
        if from_node is not to_node:  # no self loops
            if to_node not in from_node.links:  # no multi-edges
                to_nodes_stripped.append(to_node)
    sorted_indexes = list(np.argsort([abs(to_node.number_links[ie] - from_node.number_links[ie])
                                      for to_node in to_nodes_stripped]))
    to_nodes_sorted = [to_nodes_stripped[i] for i in sorted_indexes]
    running_total = 0
    stubs = np.zeros(len(to_nodes_sorted), int)
    # here we exclude self loops and multi-edges
    # and create stub lists
    for i, to_node in enumerate(to_nodes_sorted):
        stub = to_node.number_links[ie] - to_node.number_links_work[ie]
        running_total += stub
        stubs[i] = running_total

    # pick links from stubs list
    for _ in range(from_number_links):
        if stubs[-1] > 0:  # there are still nodes to assign
            jd = np.random.beta(Nodes.alfa_assortativity, Nodes.beta_assortativity)
            selected_stub = int(round(jd*(stubs[-1]-1)+1))
            selected_node = bisect_left(stubs, selected_stub)
            target_node = to_nodes_sorted[selected_node]
            drop_multi_edges = target_node.number_links[ie] - target_node.number_links_work[ie]
            stubs[selected_node:] -= drop_multi_edges  # remove selected node from the candidate list
            from_node.links[ie].append(target_node)
            from_node.number_links_work[ie] += 1
            target_node.links[ie].append(from_node)
            target_node.number_links_work[ie] += 1
        else:  # node matching failed: needs recovery
            edge_creation_complete = False

    if not edge_creation_complete:    # if list exhausted rewire another link
        # let's search for a node that is not connected to from_node
        # as this will only happen with a residual percentage of nodes let's not care about assortativity
        for to_node in np.random.permutation(to_nodes_sorted):
            number_links = len(to_node.links[ie])
            # find a node not connected to from_node
            if number_links > 0 and from_node not in to_node.links[ie] and from_node is not to_node:
                replace = np.random.randint(number_links)  # randomly select a linked node
                replaced_node = to_node.links[ie][replace]
                from_node.links[ie].append(to_node)  # add it to the list of from_node links
                from_node.number_links_work[ie] += 1
                to_node.links[ie][replace] = from_node
                # remove original linked node:
                del replaced_node.links[ie][replaced_node.links[ie].index(to_node)]
                replaced_node.number_links_work[ie] -= 1
                if DEBUG:
                    print("type ", ie, " link ", replaced_node.node_number, " to ",
                          to_node.node_number, "destroyed for ", to_node.node_number, "to",
                          from_node.node_number, " in community ", from_node.get_community().community_name)
                succeed = False
                break

    if DEBUG:
        print("assignment done ", succeed)
    difference = from_node.number_links[ie] - from_node.number_links_work[ie]
    if difference > 0:
        succeed = False
        if DEBUG:
            print("Could not connect ", difference, " stubs")
    return succeed


def build_network(communities: 'list[Community]', number_nodes, nodes: 'list[Nodes]'):
    """
    Executes the cycle to structure the network at time t

    :param communities:
    :param number_nodes:
    :param nodes:
    :return:
    """
    # computes assortativity parameters
    Nodes.alfa_assortativity = Nodes.beta_assortativity = Nodes.alfa_affinity = Nodes.beta_affinity = 1
    # compute beta_assortativity distribution parameters for assortativity requirements
    if assortativity > 0:
        Nodes.beta_assortativity = 1 + BETA_MAX * assortativity
    elif assortativity < 0:
        Nodes.alfa_assortativity = 1 + ALFA_MAX * -assortativity
    if degree_affinity > 0:
        Nodes.beta_affinity = 1 + BETA_MAX * degree_affinity
    elif degree_affinity < 0:
        Nodes.alfa_affinity = 1 + ALFA_MAX * -degree_affinity

    build_nodes(communities, number_nodes, nodes)
    for node in nodes:
        node.reset_node_links()

    generate_links_intra(communities, nodes)

    generate_links_inter(nodes)
    for node in nodes:
        node.set_lifecycle()

    print("Time step:", Ts.timestamp, "Network built with ", len(communities), "communities and ", len(nodes), "nodes.")
    assortativity_coef = compute_assortativity(compute_mixing_matrix(nodes))
    print("Assortativity Coefficient", assortativity_coef)


def compute_assortativity(mixing_matrix: 'np.array') -> 'double':
    """
    computes the Pearson correlation coefficient of mixing_matrix
    :param mixing_matrix:
    :return:
    """

    xy = np.arange(len(mixing_matrix))
    a = mixing_matrix.sum(axis=0)
    b = mixing_matrix.sum(axis=1)
    ro_a = (a * xy**2).sum() - ((a * xy).sum())**2
    ro_b = (b * xy**2).sum() - ((b * xy).sum())**2
    xy_outer = np.outer(xy, xy)
    ab_outer = np.outer(a, b)
    result = (xy_outer * (mixing_matrix - ab_outer)).sum() / np.sqrt(ro_a * ro_b)
    return result


def compute_mixing_matrix(source_nodes: 'list[Nodes]') -> 'np.array':
    highest_degree = max([sum(source_node.number_links) for source_node in source_nodes])
    mixing_matrix = np.zeros((highest_degree+1, highest_degree+1))
    total_edges = 0
    for node in source_nodes:
        degree = sum(node.number_links)
        for links in node.links:
            for link in links:
                mixing_matrix[degree, sum(link.number_links)] += 1
                total_edges += 1
    # print("4",total_edges, Nodes.total_intra, Nodes.total_degree)
    mixing_matrix = mixing_matrix / total_edges
    return mixing_matrix

