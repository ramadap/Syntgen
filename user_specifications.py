"""
User specifications go here

User can specify:
    number of time steps
    verbose output (debug)
    prefix of output files PREFIXlinks.csv, PREFIXnodes.csv
    search parameters for maximum t to t+1 similarity
    jaccard threshold for community matching
    changes to the network before transition from t to t+1c (deletion of nodes)
    community size distribution (can be changed at any step=
    node degree distribution (can be changed at any step)
    whether nodes should try to keep affinity (i.e. a popular node should remain popular)
    whether to normalize the jaccard index to a null random model
    what to print



"""
import numpy as np
from random import sample


# User specified constants
TIMESTEPS = 5
DEBUG = False                       # if detailed printouts are required
name_of_output_file = 'graph'

# positive= assortative, negative= dissortative, zero= no joint degree distribution
degree_affinity = 0                 # affinity of node degrees across intervals (random if 0, [-1,1])
assortativity = 0                   # degree of dis/assortativity ( -1 = dissortative, 1 = assortative)

jaccard_null_model = True             # using jaccard adjusted to null random flow (normal jaccard if false)
sigma = 0.2                           # jaccard threshold for community matching

stop_if_disjoint = False              # Abort run if community is disjoint
retry_new_degree_distribution = 10    # Number of times to retry node links distributions in case dist non-graphable

write_gephi = True                    # write file for gephi
write_netgram = False                 # write file for matlab


def search_parameters() -> 'int, int, int, int, int':
    """ returns parameters to control the search for an optimal solution

    invoked at every timestep ( available globally at Ts.timestamp )

    local_search: maximum number of greedy searches on a single strand
    drop_local_search: maximum number of greedy searches without improvement
    global_search: maximum number of greedy searches restarts from best so far
    drop_global_search: maximum number of greedy searches restarts without improvement
    search_type: base starts:
                    1: try all basic initial algos
                    2: try only the best
                    3: use only the result from algo (recommended for networks with more than 20-30 communities)

    :return: parameters
    """

    local_search = 1000
    drop_local_search = 30
    global_search = 200
    drop_global_search = 2
    search_type = 2

    return local_search, drop_local_search, global_search, drop_global_search, search_type


def print_parameters():
    """
    Return booleans to control print output at the end of each snapshot
    :return: confusion_matrix_print, confusion_matrix_percentage, jaccard_index, continuity, \
           community_events_t0, community_events_t1
    """
    confusion_matrix_print = True
    confusion_matrix_percentage = True
    jaccard_index = True
    continuity = True
    community_events_t0 = True
    community_events_t1 = True
    return confusion_matrix_print, confusion_matrix_percentage, jaccard_index, continuity, \
        community_events_t0, community_events_t1


# @test_changes_integrity
def user_changes_specs(communities: 'list[Community]', nodes: 'list[Nodes]') -> 'list[Nodes]':
    """ returns a list of nodes to delete. it's up to the user which nodes should be killed
    :param communities: list of community objects
    :param nodes: list of node objects
    :return: dead_node_vector: list of nodes to delete
    """
    kill = 0.1
    nodes_to_delete = round(kill*len(nodes))                                 # set elimination target
    dead_node_vector = sample(nodes, nodes_to_delete)
    return dead_node_vector


def power_law(k_min, k_max, gamma):
    return ((k_max**(-gamma+1) - k_min**(-gamma+1))*np.random.random() + k_min**(-gamma+1.0))**(1.0/(-gamma + 1.0))


def community_distribution_power_law() -> 'list[int]':
    """ returns a community size distribution in a list

    In this example a power law distribution according to default parameters is returned. User is free to code it's own
    distribution.


    :return: list of community sizes
    """

    # example if a change in the network structure is required during the run
    # time_series = {'desired_number_nodes': [400, 1000],
    #              'delta': [4][5],
    #              'max_community_size': [200, 600],
    #              'min_community_size': [20, 30]}
    # desired_number_nodes = time_series['desired_number_nodes'][Ts.timestamp]
    # delta = time_series['delta'][Ts.timestamp]
    # max_community_size = time_series['max_community_size'][Ts.timestamp]
    # min_community_size = time_series['min_community_size'][Ts.timestamp]

    # :param desired_number_nodes: Approximate number of nodes requested
    # :param delta: power law exponent
    # :param max_community_size: size of the largest community
    # :param min_community_size: size of the smallest community
    delta = 1.5
    desired_number_nodes = 1000
    max_community_size = 200
    min_community_size = 30

    number_communities = 0
    generated_nodes = 0
    community_sizes = []
    while generated_nodes < desired_number_nodes:
        rc = int(power_law(min_community_size, max_community_size, delta))
        community_sizes.append(rc)
        generated_nodes += rc
        number_communities += 1
    if generated_nodes - desired_number_nodes > rc / 2:
        community_sizes = community_sizes[:-1].copy()
        number_communities -= 1
    return community_sizes


def node_distribution_power_law(community_sizes: 'list[int]', retries) -> 'list[int],list[int]':
    """
    returns two node degree distributions: total  and INTRA

    to generate feasible distributions there should not be a skew towards large and small degrees (bathtub)
    maximum degree should be substantially lower than community size???
    :param community_sizes: Community sizes distribution
    :param retries:  retry number if previous sequence non graphic

    :return: lists of total and INTRA node degrees
    """

    # :param gamma: power law exponent
    # :param max_degree: maximum node degree
    # :param min_degree:  minimum node degree
    # :param mix_ratio: ratio of INTRA to TOTAL
    # :param fixed: whether the ratio is fixed or IID randomly distributed around the mix_ratio
    gamma = 2.5
    max_degree = 40
    min_degree = 8
    mix_ratio = .7
    fixed = False

    number_nodes = sum(community_sizes)
    largest_community_size = max(community_sizes)
    if fixed:
        limit_degree = int(largest_community_size / mix_ratio)
    else:
        limit_degree = largest_community_size
    if limit_degree <= max_degree:
        max_degree = limit_degree - 1

    node_degrees = []
    node_intra_degree = []
    for i in range(number_nodes):
        rc = int(power_law(min_degree, max_degree, gamma))
        node_degrees.append(rc)
        if fixed:
            degree = rc * mix_ratio
            if np.random.random() < degree % 1:
                node_intra_degree.append(int(np.ceil(degree)))
            else:
                node_intra_degree.append(int(np.floor(degree)))
        else:
            node_intra_degree.append(np.random.binomial(rc, mix_ratio))

    return node_degrees, node_intra_degree


def community_distribution_random() -> 'list[int]':
    desired_number_nodes = 1000
    max_community_size = 200
    min_community_size = 30

    number_communities = 0
    generated_nodes = 0
    community_sizes = []
    while generated_nodes < desired_number_nodes:
        rc = int(round(min_community_size + np.random.random() * (max_community_size - min_community_size)))
        community_sizes.append(rc)
        generated_nodes += rc
        number_communities += 1
    if generated_nodes - desired_number_nodes > rc / 2:
        community_sizes = community_sizes[:-1].copy()
        number_communities -= 1
    return community_sizes


def node_distribution_random(community_sizes: 'list[int]', retries) -> 'list[int],list[int]':
    """
    returns two node degree distributions: TOTAL and INTRA

    to generate feasible distributions there should not be a skew towards large and small degrees (bathtub)
    maximum degree should be substantially lower than community size???
    :param community_sizes: distribution of community sizes
    :param retries:  retry number if previous sequence non graphic

    :return: lists of TOTAL and INTRA node degrees
    """

    # :param gamma: power law exponent
    # :param max_degree: maximum node degree
    # :param min_degree:  minimum node degree
    # :param mix_ratio: ratio of INTRA to INTER
    # :param fixed: whether the ratio is fixed or IID randomly distributed around the mix_ratio
    pkk = .2        # probability of an intra link
    pkn = .002       # probability of an inter link
    mix_ratio = .7  # ratio of intra to total links (for fixed = True)
    fixed = False    # if ratio of intra to total is fixed (ala LFR), must specify mix_ratio

    number_nodes = sum(community_sizes)

    node_intra_degree = []
    node_total_degree = []

    for community_size in community_sizes:
        for i in range(community_size):
            degree = np.random.binomial(community_size-2, pkk) + 1  # we want at least one link
            node_intra_degree.append(degree)
            if fixed:
                node_total_degree.append(int(round(degree / mix_ratio)))
            else:
                node_total_degree.append(np.random.binomial(number_nodes-community_size, pkn))
                node_total_degree[-1] += node_intra_degree[-1]

    return node_total_degree, node_intra_degree


def community_distribution_exponential() -> 'list[int]':
    """ returns a community size distribution in a list

    In this example an exponential distribution according to default parameters is returned. User is free to code
    it's own distribution.


    :return: list of community sizes
    """

    # example if a change in the network structure is required during the run
    # time_series = {'desired_number_nodes': [400, 1000],
    #              'delta': [4][5],
    #              'max_community_size': [200, 600],
    #              'min_community_size': [20, 30]}
    # desired_number_nodes = time_series['desired_number_nodes'][Ts.timestamp]
    # delta = time_series['delta'][Ts.timestamp]
    # max_community_size = time_series['max_community_size'][Ts.timestamp]
    # min_community_size = time_series['min_community_size'][Ts.timestamp]

    # :param desired_number_nodes: Approximate number of nodes requested
    # :param delta: power law exponent
    # :param max_community_size: size of the largest community
    # :param min_community_size: size of the smallest community
    beta = 1                    # scale parameter and mean
    desired_number_nodes = 1000
    max_community_size = 200
    min_community_size = 30

    exp_dist = np.random.exponential(beta, int(np.ceil(desired_number_nodes/min_community_size)))
    max_exp_dist = max(exp_dist)
    min_exp_dist = min(exp_dist)
    dif_community_size = max_community_size - min_community_size
    dif_exp_dist = max_exp_dist - min_exp_dist
    continue_search = True
    while continue_search:
        i = 0
        number_communities = 0
        generated_nodes = 0
        community_sizes = []
        while generated_nodes < desired_number_nodes:
            rc = int(round((exp_dist[i]-min_exp_dist) * dif_community_size / dif_exp_dist)) + min_community_size
            community_sizes.append(rc)
            generated_nodes += rc
            number_communities += 1
            i += 1

        if generated_nodes - desired_number_nodes > rc / 2:
            community_sizes = community_sizes[:-1].copy()
            number_communities -= 1
            i -= 1
        if (max(exp_dist[0:i]), min(exp_dist[0:i])) == (max_exp_dist, min_exp_dist):
            continue_search = False
        else:                                   # try again
            max_exp_dist = max(exp_dist[0:i])
            min_exp_dist = min(exp_dist[0:i])
            dif_exp_dist = max_exp_dist - min_exp_dist

    return community_sizes


def node_distribution_exponential(community_sizes: 'list[int]', retries) -> 'list[int],list[int]':
    """
    returns two node degree distributions: total  and INTRA

    to generate feasible distributions there should not be a skew towards large and small degrees (bathtub)
    maximum degree should be substantially lower than community size???
    :param community_sizes: Community sizes distribution
    :param retries:  retry number if previous sequence non graphic

    :return: lists of total and INTRA node degrees
    """

    # :param gamma: power law exponent
    # :param max_degree: maximum node degree
    # :param min_degree:  minimum node degree
    # :param mix_ratio: ratio of INTRA to INTER
    # :param fixed: whether the ratio is fixed or bernoulli distributed around the mix_ratio
    beta = 1                # inverse of the rate parameter
    max_degree = 40
    min_degree = 8
    mix_ratio = .7
    fixed = False

    number_nodes = sum(community_sizes)
    largest_community_size = max(community_sizes)
    if fixed:
        limit_degree = int(largest_community_size / mix_ratio)
    else:
        limit_degree = largest_community_size
    if limit_degree <= max_degree:
        max_degree = limit_degree - 1

    node_degrees = []
    node_intra_degree = []
    exp_dist = np.random.exponential(beta, number_nodes)
    min_exp_dist = min(exp_dist)
    dif_exp_dist = max(exp_dist) - min_exp_dist

    for i in range(number_nodes):
        rc = int(round(((exp_dist[i]-min_exp_dist) * (max_degree - min_degree) / dif_exp_dist) + min_degree))
        node_degrees.append(rc)
        if fixed:
            node_intra_degree.append(int(round(rc * mix_ratio)))
        else:
            node_intra_degree.append(np.random.binomial(rc, mix_ratio))

    return node_degrees, node_intra_degree


community_distribution = community_distribution_power_law
node_distribution = node_distribution_power_law
user_changes = user_changes_specs
