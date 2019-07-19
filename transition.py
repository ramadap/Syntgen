"""
This module aggregates all objects, methods and functions related to the temporal aspects of a dynamic network

It includes the logic to determine node flow between successive time intervals
"""

import sys
import itertools as it
import numpy as np
from constants import *
from network import Community, Nodes, create_new_nodes
from user_specifications import DEBUG, search_parameters, user_changes


class State:

    def __init__(self, source, target, how_similar):
        self.source = source  # [list of source communities cardinalities
        self.target = target  # [list of target communities cardinalities
        self.source_len = len(source)  # number of source communities
        self.target_len = len(target)  # number of target communities
        self.edges_len = self.source_len * self.target_len
        self.total_nodes = sum(source)  # total number of nodes
        assert self.total_nodes == sum(target)
        # function to return similitude, receives a state object returns a real number, inversely proportional
        self.how_similar = how_similar
        self.vmax = np.zeros((self.source_len, self.target_len), int)  # higher boundary for solution
        for i in range(self.source_len):
            for j in range(self.target_len):
                self.vmax[i, j] = min(source[i], target[j])  # compute higher boundary
        self.v = np.reshape(self.vmax, self.edges_len)
        self.edges = self.floor = np.zeros((self.source_len, self.target_len), int)
        self.state = np.reshape(self.edges, self.edges_len)
        self.tabu = set()  # visited sites
        self.similarity = 0

    def set_state(self, state):
        self.state[:] = np.array(state)[:]
        self.similarity = self.how_similar(self)

    def move(self):
        s_cycle = it.combinations(range(self.source_len), 2)  # source iterator for simple cycles
        dif = self.v - self.state  # maximum increase
        #        self.edges = self.floor = np.reshape(self.state, (self.source_len, self.target_len))
        headroom = np.reshape(dif, (self.source_len, self.target_len))  # matrix view
        curr_vi_max = sys.float_info.max
        max_non_extreme = np.count_nonzero(self.state) - (len(self.state) - np.count_nonzero(dif))  # count extrema
        for x in s_cycle:
            t_cycle = it.combinations(range(self.target_len), 2)  # target iterator for simple cycles
            for y in t_cycle:
                for i in [0, 1]:  # flip flop: test for increases and decreases
                    yt = y[i], y[1 - i]
                    yr = y[1 - i], y[i]
                    if all([*self.floor[x, yr], *headroom[x, yt]]):  # is it possible?
                        max_increment = min(*self.floor[x, yr], *headroom[x, yt])  # how much?
                        save_state = self.edges[x, yt]  # save and apply
                        save_state_rev = self.edges[x, yr]
                        self.edges[x, yt] += max_increment
                        self.edges[x, yr] -= max_increment
                        #  print(max_negative_increment, "-", state)
                        if tuple(self.state) not in self.tabu:  # already visited?
                            non_extreme_points = np.count_nonzero(self.state) - (
                                        len(self.state) - np.count_nonzero(dif))
                            if non_extreme_points < max_non_extreme:  # if more on edge, select
                                # print("done-", non_extreme_points, max_non_extreme)
                                max_non_extreme = non_extreme_points
                                best_state = self.state.copy()
                                curr_vi_max = self.how_similar(self)
                            elif non_extreme_points == max_non_extreme:  # otherwise select depending on
                                varinf = self.how_similar(self)  # metric
                                if varinf < curr_vi_max:
                                    if varinf != self.similarity:  # avoid trivial changes
                                        curr_vi_max = varinf
                                        best_state = self.state.copy()
                                        # print(best_state, curr_vi_max)

                        self.edges[x, yt] = save_state  # reset for next iteration
                        self.edges[x, yr] = save_state_rev

        #    print("in move: ",state,edges,curr_rand_max, best_state)
        self.similarity = sys.float_info.max  # assume dead end
        if curr_vi_max < sys.float_info.max:  # if not, return best
            self.tabu.add(tuple(best_state))
            self.state[:] = best_state[:]
            self.similarity = curr_vi_max

        return self.similarity, self.state


def events_communities(communities_t0: 'list[Community]', nodes_t0: 'list[Nodes]', communities_t1: 'list[Community]',
                       nodes_t1: 'list[Nodes]', number_nodes_t1: 'int') -> 'np.array(int, int)':
    """
    Codes all community events between successive time steps.

    Community events are driven by the differences between communities at successive time steps.
    The differences stem from user initiated node deletion at time t, the network size, community size and node degree
    distributions as specified by the user, and the matching process between t and t+1 as a result fo the randomness
    of the generating model.
    :param communities_t0:
    :param nodes_t0:
    :param communities_t1:
    :param nodes_t1:
    :param number_nodes_t1:
    :return:
    """

    # note that a community can get destroyed by node depletion at time t0. Need to use original cardinality.
    number_communities_t0 = len(communities_t0)
    number_communities_t1 = len(communities_t1)
    confusion_matrix = np.zeros([number_communities_t0 + 1, number_communities_t1 + 1])
    # include code for events over t0
    dead_node_vector = user_changes(communities_t0, nodes_t0)
    number_of_deaths = 0
    for node in dead_node_vector:
        confusion_matrix[node.get_community().get_community_number(), -1] += 1
        node.kill_node(nodes_t0)
        number_of_deaths += 1

    nodes_difference = number_nodes_t1 - len(nodes_t0)

    if nodes_difference == 0:
        return confusion_matrix

    if nodes_difference > 0:  # network grew
        # network size increase: create new nodes_t0 per community
        if DEBUG:
            print("network grew by", nodes_difference)
        dif = 0
        while dif < nodes_difference:  # do until difference is nullified
            n = 0
            ix = np.random.randint(number_nodes_t1)  # select community with probability  f(size)
            for community_t1 in communities_t1:
                n += community_t1.total_nodes  # accumulate number of nodes_t0
                if ix < n:
                    if community_t1.get_number_nodes() < community_t1.total_nodes:  # ensure max  not exceeded
                        create_new_nodes(community_t1, nodes_t1, 1)
                        # updates "phantom" com. of new nodes_t0
                        confusion_matrix[-1, community_t1.community_number] += 1
                        dif += 1
                    break

    else:  # network shrank
        # network size decrease remove nodes_t0 from t0 network
        if DEBUG:
            print("network shrank by", -nodes_difference)
        number_of_deaths -= nodes_difference
        for _ in range(-nodes_difference):
            ix = np.random.randint(len(nodes_t0))
            # updates "phantom" com. of dead nodes_t0
            confusion_matrix[nodes_t0[ix].community.get_community_number(), -1] += 1
            nodes_t0[ix].kill_node(nodes_t0)
    if number_of_deaths > 0 or nodes_difference > 0:
        print("At end of time step:", Ts.timestamp - 1, ",", )
    if number_of_deaths > 0:
        print(number_of_deaths, " nodes died", end="")
    if nodes_difference > 0:
        print(" and", nodes_difference, " nodes were born")
    else:
        print()
    if number_communities_t0 != number_communities_t1:
        print("number of communities changed from", len(communities_t0), "to", len(communities_t1))

    return confusion_matrix


def transition_t0_t1(communities_t0: 'list[Community]', communities_t1: 'list[Community]') -> 'float, list[int]':
    """
    Optimizes the flow of nodes from tne network at time t to time t+1 so that the resulting communities are the
    most similar possible.

    Starts from a good-enough solution. search_type in "constants" controls the search:
    Uses best of good enough solutions if search_type == 3
    Starts from the best of good enough solutions if search_type == 2
    searches from all good enough unique solutions (in algo soup) if search_type == 3

    :param communities_t0:
    :param communities_t1:
    :return: score (how similar the partitionings are) and flow list
    """
    local_search, drop_local_search, global_search, drop_global_search, search_type = search_parameters()
    source = [community.get_number_nodes() for community in communities_t0]
    # do not consider new nodes
    target = [community.total_nodes - community.get_number_nodes() for community in communities_t1]
    # print(source, [community.get_number_nodes() for community in communities_t1], target)
    state_algos = []

    for algo in soup:
        state_algos += [State(source, target, vi)]
        state_algos[-1].set_state(algo(state_algos[-1]))
        print ("VI=", state_algos[-1].similarity, " Node Flow:", state_algos[-1].state)
    if DEBUG:
        [print(state_algos.similarity) for state_algos in state_algos]

    state_algos.sort(key=lambda x: x.similarity)

    # remove duplicate initial solutions
    j = 1
    while j < len(state_algos):
        if (state_algos[j].state == state_algos[j-1].state).all():
            del state_algos[j]
            if DEBUG:
                print("duplicate starting point, deleted")
        else:
            j += 1

    if DEBUG:
        for state in state_algos:
            print(state.similarity, state.state)
    best_global = state_algos[0].similarity
    best_global_solution = state_algos[0].state
    if search_type == 3:
        return best_global, best_global_solution
    if search_type == 2:
        del state_algos[1:]
    tabu_copy = set()
    for state in state_algos:
        state.tabu = tabu_copy                                      # don't lose taboos from previous state
        global_failed = 0
        best_local_solution = state.state
        best_local = state.similarity
        print("Best solution without search", best_local)

        for i in range(global_search):
            failed = 0
            last_found = 0
            if DEBUG:
                print("new local search", best_local_solution)
            state.set_state(best_local_solution)
            best_local = state.similarity

            for j in range(local_search):
                #       t1 = time.time()
                test_best, sol = state.move()
                #       print(time.time() - t1)
                #       if test_best >= best_local: print(test_best, sol)
                if test_best == sys.float_info.max:
                    if DEBUG:
                        print(" dead end @ ", j, ", last_found @", last_found)
                    global_failed = drop_global_search
                    break
                if test_best == best_local:
                    if DEBUG:
                        print(test_best, "=")
                elif test_best < best_local:
                    best_local = test_best
                    best_local_solution = sol.copy()
                    if test_best < best_global:
                        print(test_best, "++")
                        last_found = j
                        best_global = test_best
                        best_global_solution = sol.copy()
                        failed = 0
                    else:
                        if DEBUG:
                            print(test_best, "+")
                else:
                    if DEBUG:
                        print(test_best, "-")
                    failed += 1
                if failed == drop_local_search:
                    if DEBUG:
                        print('failed to improve during local search @ ', j, '/', i, ', last found:',
                            last_found, ', bg ', best_global)
                    break
            #       if best_local == best_global:
            #            best_global_solution += [best_local_solution.copy()]
            #            print("added:", len(best_local_solution))
            if last_found > 0:
                print('new global ', best_global, best_global_solution)
                global_failed = 0
            else:
                global_failed += 1
            if global_failed >= drop_global_search:
                if DEBUG:
                    print('global failure to improve ', global_failed, best_global, best_global_solution)
                break
        tabu_copy = state.tabu

    if DEBUG:
        print("adopted solution: ", best_global, best_global_solution)
    return best_global, best_global_solution


def transfer_nodes(source_community, target_community, nodes_t1, number_nodes):
    assert target_community.total_nodes >= target_community.get_number_nodes() + number_nodes
    assert source_community.get_number_nodes() >= number_nodes
    for i in range(number_nodes):
        source_community.nodes[-1].change_community(target_community)
        # resets links as no longer relevant, but keeps number of links in case needed for new distribution
        source_community.nodes[-1].links = [[], []]
        node_to_transfer = source_community.nodes.pop(-1)
        target_community.nodes.append(node_to_transfer)
        nodes_t1.append(node_to_transfer)


def flow_nodes(communities_t0, communities_t1, nodes_t1, solution, confusion_matrix):
    # Routine to assign nodes
    solution = list(solution)
    for source_community in communities_t0:
        for target_community in communities_t1:
            flow = solution.pop(0)
            confusion_matrix[source_community.get_community_number(), target_community.get_community_number()] += flow
            transfer_nodes(source_community, target_community, nodes_t1, flow)


def match_communities(communities_t0: 'list[Community]', communities_t1: 'list[Community]',
                      confusion_matrix: 'np.array(int,int)', sigma: 'float', jaccard_null_model: 'bool'):
    """
    Match communities between T and T+1

    :param communities_t0: community objects at time T
    :param communities_t1: Community objects at time T+1
    :param confusion_matrix: Matrix of shared nodes between communities @ time T & T+1
    :param sigma: Jaccard index threshold
    :param jaccard_null_model; True if jaccard should be normalized to [random null - 1]
    :return:
    """

    # sizes of all communities
    t0_sizes = confusion_matrix.sum(axis=1)
    t1_sizes = confusion_matrix.sum(axis=0)

    # build the matrix of the union cardinalities for the communities cartesian product
    t0_transpose = np.array([t0_sizes]).transpose()
    t_union = t1_sizes + t0_transpose

    # compute the jaccard index
    # avoid 0/0 errors for no births and deaths
    with np.errstate(invalid='ignore'):
        Community.jaccard_index = confusion_matrix / (t_union - confusion_matrix)
        if np.isnan(Community.jaccard_index[-1, -1]):
            Community.jaccard_index[-1, -1] = 0

    # if the user wants it normalized to the null hypothesis
    if jaccard_null_model:
        t1_total = t1_sizes.sum()
        confusion_null = t1_sizes / t1_total * t0_transpose
        # avoid 0/0 errors for no births and deaths
        with np.errstate(invalid='ignore'):
            jaccard_null = confusion_null / (t_union - confusion_null)
            if np.isnan(jaccard_null[-1, -1]):
                jaccard_null[-1, -1] = 0
        Community.jaccard_index = (Community.jaccard_index - jaccard_null) / (1 - jaccard_null)
        Community.jaccard_index[Community.jaccard_index < 0] = 0

    # compute selection criteria 1: if jaccard index above sigma, 0: otherwise
    # exclude births and deaths
    Community.continuation = Community.jaccard_index[0:-1, 0:-1].copy()
    Community.continuation[Community.continuation < sigma] = 0
    Community.continuation[Community.continuation > 0] = 1

    # compute number of successful relations per community (splits on x axis, merges on y axis)
    splits = Community.continuation.sum(axis=1)
    merges = Community.continuation.sum(axis=0)
    # indexes where flow of nodes exist
    t0 = np.where(splits == 1)[0]  # indexes of relations for t0
    t1 = np.where(merges == 1)[0]

    # compute all entries that contain a single 1 in the continuation row/column matrix
    single_flows = list(it.product(t0, t1))

    # and build tuples of all possible entries
    continuation_indexes = list(zip(*np.where(Community.continuation == 1)))

    # now fnd all continuations
    [(communities_t0[t[0]].set_ground_truth("Continues " + ("growing" if t0_sizes[t[0]] < t1_sizes[t[1]] else
                                                            "shrinking" if t0_sizes[t[0]] > t1_sizes[t[1]] else "")
                                            + " in", communities_t1[t[1]]),
      communities_t1[t[1]].set_ground_truth("Continued " + ("growing" if t0_sizes[t[0]] < t1_sizes[t[1]] else
                                                            "shrinking" if t0_sizes[t[0]] > t1_sizes[t[1]] else "")
                                            + " from", communities_t0[t[0]]),
      communities_t1[t[1]].set_community_name(communities_t0[t[0]].get_community_name()))
     for t in continuation_indexes if t in single_flows]

    # find deaths
    [communities_t0[i].set_ground_truth("Died", [])
     for i in np.where(splits == 0)[0]]

    # find splits
    [communities_t0[i].set_ground_truth("Split into",
                                        [communities_t1[j] for j in np.where(Community.continuation[i] == 1)[0]])
     for i in np.where(splits > 1)[0]]

    # find merges
    [communities_t0[t[0]].set_ground_truth("Merged Into", communities_t1[t[1]])
     for t in continuation_indexes if t not in single_flows and merges[t[1]] > 1]

    # find births
    [communities_t1[i].set_ground_truth("Born", []) for i in np.where(merges == 0)[0]]

    # birth from split
    [(communities_t1[t[1]].set_ground_truth("from Split", communities_t0[t[0]]))
     for t in continuation_indexes if t not in single_flows and splits[t[0]] > 1]

    # birth from merge
    [communities_t1[i].set_ground_truth("Merged from",
                                        [communities_t0[j] for j in np.where(Community.continuation[:, i])[0]])
     for i in np.where(merges > 1)[0]]


def turnover_communities(communities_from, nodes_from):

    Community.highest_community_number = 0
    # reset ground truth
    [community.set_ground_truth('reset', []) for community in communities_from]
    return communities_from, nodes_from


def vi(state):
    """
    computes variation of information between two partitionings represented in "state"
    :param state: state object with the bipartite network flow
    :return: variation of information (the closest to 0, the more similar )
    """

    vi = 0
    for i in range(state.source_len):
        for j in range(state.target_len):
            if state.edges[i, j] > 0:
                rij = state.edges[i, j] / state.total_nodes
                vi -= rij * np.log2(state.edges[i, j] * state.edges[i, j] / (state.source[i] * state.target[j]))
    return round(vi, 12)


def assign_in_order(state: 'State') -> 'np.ndarray':
    """
    Assigns nodes between largest clusters in size order, reinjecting the remainder into cluster list

    Simple heuristic to optimize variation of information between two partitionings
    :param state:
    :return: a solution
    """
    targetw = list(state.target)
    sourcew = list(state.source)
    sol = np.zeros(state.edges_len, int)
    index_source = list(reversed(np.argsort(sourcew)))
    index_target = list(reversed(np.argsort(targetw)))
    complete = False
    while not complete:
        if sourcew[index_source[0]] == 0:
            complete = True
        else:
            if sourcew[index_source[0]] <= targetw[index_target[0]]:
                sol[index_source[0] * state.target_len + index_target[0]] = sourcew[index_source[0]]
                targetw[index_target[0]] -= sourcew[index_source[0]]
                sourcew[index_source[0]] = 0
            else:
                sol[index_source[0] * state.target_len + index_target[0]] = targetw[index_target[0]]
                sourcew[index_source[0]] -= targetw[index_target[0]]
                targetw[index_target[0]] = 0
            index_source = list(reversed(np.argsort(sourcew)))
            index_target = list(reversed(np.argsort(targetw)))
    return sol


def closest(state):
    """
    Assigns nodes between clusters by increasing size difference, reinjects the remainder into the lists
    :param state:
    :return:
    """
    sol = np.zeros(state.edges_len, int)
    targetw = list(state.target)
    sourcew = list(state.source)
    complete = False
    while not complete:
        if sum(sourcew) == 0:
            complete = True
        else:
            maxdif = max([max(sourcew), max(targetw)])
            for j in range(state.source_len):
                for i in range(state.target_len):
                    if sourcew[j] > 0 and targetw[i] > 0:
                        if abs(targetw[i] - sourcew[j]) < maxdif:
                            maxdif = abs(targetw[i] - sourcew[j])
                            ii = i
                            jj = j
            if sourcew[jj] <= targetw[ii]:
                sol[jj * state.target_len + ii] = sourcew[jj]
                targetw[ii] -= sourcew[jj]
                sourcew[jj] = 0
            else:
                sol[jj * state.target_len + ii] = targetw[ii]
                sourcew[jj] -= targetw[ii]
                targetw[ii] = 0
    #        print(ii,jj,sourcew, targetw)

    return sol


def minstep_jaccard(state):
    """
    Assigns nodes between clusters by increasing contribution to a global jaccard index
    :param state:
    :return:
    """
    sol = np.zeros(state.edges_len, int)
    targetw = list(state.target)
    sourcew = list(state.source)
    complete = False
    while not complete:
        if sum(sourcew) == 0:
            complete = True
        else:
            while not complete:
                if sum(sourcew) == 0:
                    complete = True
                else:
                    maxdif = -1
                    for j in range(state.source_len):
                        for i in range(state.target_len):
                            if sourcew[j] > 0 and targetw[i] > 0:
                                dif = min(targetw[i], sourcew[j]) / max(targetw[i], sourcew[j])
                                if dif > maxdif:
                                    maxdif = dif
                                    ii = i
                                    jj = j
                    if sourcew[jj] <= targetw[ii]:
                        sol[jj * state.target_len + ii] = sourcew[jj]
                        targetw[ii] -= sourcew[jj]
                        sourcew[jj] = 0

                    else:
                        sol[jj * state.target_len + ii] = targetw[ii]
                        sourcew[jj] -= targetw[ii]
                        targetw[ii] = 0

    return sol


def minstep_vi(state):
    """
    assigns nodes between clusters by increasing contribution to a global variation of information
    :param state:
    :return:
    """
    sol = np.zeros(state.edges_len, int)
    targetw = list(state.target)
    sourcew = list(state.source)
    complete = False
    while not complete:
        if sum(sourcew) == 0:
            complete = True
        else:
            maxdif = sys.float_info.max
            for j in range(state.source_len):
                for i in range(state.target_len):
                    if sourcew[j] > 0 and targetw[i] > 0:
                        lowest = min(targetw[i], sourcew[j])
                        rij = lowest / state.total_nodes
                        dif = -rij * np.log2(lowest * lowest / (sourcew[j] * targetw[i]))
                        if dif < maxdif:
                            maxdif = dif
                            ii = i
                            jj = j
            if sourcew[jj] <= targetw[ii]:
                sol[jj * state.target_len + ii] = sourcew[jj]
                targetw[ii] -= sourcew[jj]
                sourcew[jj] = 0
            else:
                sol[jj * state.target_len + ii] = targetw[ii]
                sourcew[jj] -= targetw[ii]
                targetw[ii] = 0

    return sol


def minstep_mi(state):
    """
    assigns nodes between clusters by increasing contribution to a global mutual information
    :param state:
    :return:
    """
    sol = np.zeros(state.edges_len, int)
    targetw = list(state.target)
    sourcew = list(state.source)
    complete = False
    while not complete:
        if sum(sourcew) == 0:
            complete = True
        else:
            maxdif = -1
            for j in range(state.source_len):
                for i in range(state.target_len):
                    if sourcew[j] > 0 and targetw[i] > 0:
                        lowest = min(targetw[i], sourcew[j])
                        rij = lowest / state.total_nodes
                        dif = rij * np.log2(rij / (sourcew[j] * targetw[i] / (state.total_nodes * state.total_nodes)))
                        if dif > maxdif:
                            maxdif = dif
                            ii = i
                            jj = j
            if sourcew[jj] <= targetw[ii]:
                sol[jj * state.target_len + ii] = sourcew[jj]
                targetw[ii] -= sourcew[jj]
                sourcew[jj] = 0
            else:
                sol[jj * state.target_len + ii] = targetw[ii]
                sourcew[jj] -= targetw[ii]
                targetw[ii] = 0

    return sol


# these are the algorithms implementing simple heuristics to find good_enough solutions for initial searches or as a
# substitute to those searches when the problem becomes intractable due to size.
soup = [assign_in_order, closest, minstep_jaccard, minstep_vi, minstep_mi]
