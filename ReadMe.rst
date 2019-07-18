**************************************************************************************
Syntgen: A synthetic temporal network generator with clustering and known ground truth
**************************************************************************************

Description
#############
Please refer to "Syntgen: A system to generate temporal networks with user specified topology", submitted to IMA Journal
of Complex Networks, for network generation.
Please refer to "A Taxonomy of Community Lifecycle Events in Temporal Networks", on IEEE Xplore repository (TBD),
for lifecycle event classification

This package is a Python system to generate discrete temporal, non weighted, non overlapped, non directional networks, that can exhibit community structure. 
The system generates a network according to user specifications, linking nodes randomly or subject to degree assortativity specifications, with user supplied sequences of community sizes, and bijections of nodes total and intra-community degrees. Power-law and other distribution samplers are provided as examples of sequences generators. The system attempts to keep the minimum shared information distance between clusterings across successive time transitions. Sequences can be changed by the user at every transition, and selected nodes removed.

Possible use cases:
===================

1. Test and benchmark algorithms that operate on temporal networks, such as community detection and evolution algorithms;
2. Experimentally study correlation of network metrics;
3. Analyse empiric networks under varying network properties (by loading community sequences, node degrees and node
   lifecycle data from empiric temporal networks).

Pre-requisites
==============
Python v3.5+
Libraries: numpy, itertools, random, bisect
Gephi (if needed for visualization)
Netgram and (Matlab or Octave) (if needed for visualization)


Operation
==========
to run:
    execute syntgen


Input:
    User parameters described below


Output:
	Console text describing network metrics, events and node flow

	CSV file with time tagged node links (formatted to be imported into the Gephi package)

	CSV file with time tagged node community membership	 (formatted to be imported into the Gephi package)

	TXT file with time tuples of nodes and community attribution 	(formatted for Netgram, read by loading
	Netgram.m Matlab/Octave script )


Parameters
=============

All input specified in file **user_specifications.py**

.. csv-table:: Behaviour parameters
   :header: "Parameter", "Description", "Default"
   :widths: 15, 100, 10

    "TIMESTEPS",number of time steps: Run duration in timesteps,5
    "DEBUG",verbose output (debug),No
    "degree_affinity","nodes should try to keep degree temporal affinity (0 if random) range [-1 to 1])",0
    "Assortativity","whether the system should try to maintain node degree assortativity (range -1 dissortative to 1 assortative), structural cutoffs not withstanding",0
    "jaccard_null_model",whether to adjust the jaccard index to a null random model when comparing communities for lifecycle determination",TRUE
    "name_of_output_file","prefix of output files PREFIXlinks.csv, PREFIXnodes.csv, PREFIXnodes.txt","""graph"""
    "write_gephi","produces two csv files to be loaded into gephi for visualization",TRUE
    "write_netgram","produces one text file to be loaded into netgram for visualization",FALSE
    "Sigma","jaccard threshold for community matching, value between [0,1] 1 requires exact matching",0.2
    "stop_if_disjoint","Abort the run if a community is disjoint",FALSE
    "retry_new_degree_distribution","number of times a new node degree sequence is requested per time step if a provided sequence is non graphic",5
    "user_changes","function to delete nodes between snapshots","user_changes_specs"
    "community_distribution","function to return a community size sequence for snapshot creation","community_distribution_power_law"
    "node_distribution","function to return bijection of total,intra node degrees","node_distribution_power_law"


User defined functions
***********************

Setting parameters for minimum VI search across snapshots
*********************************************************
.. code:: python

    def search_parameters() -> 'int, int, int, int, int':
        """ returns parameters to control the search for an optimal solution

        invoked at every timestep ( available globally at Ts.timestamp )

        local_search: maximum number of greedy searches on a single strand (default 1000)
        drop_local_search: maximum number of greedy searches without improvement (default 30)
        global_search: maximum number of greedy searches restarts from "best so far" (default 200)
        drop_global_search: maximum number of greedy searches restarts without improvement (default 2)
        search_type: base starts:
                        1: try all basic initial algos
                        2: try only the best    (default)
                        3: use only the result from algo (recommended for networks with more than 20-30 communities)

        :return: parameters
        """

Print parameters
*********************************************************
.. code:: python

    def print_parameters():
        """
        Return booleans to control print output at the end of each snapshot
        :return: confusion_matrix_print, confusion_matrix_percentage, jaccard_index, continuity, \
               community_events_t0, community_events_t1
    # Defaults
    confusion_matrix_print = True
    confusion_matrix_percentage = True
    jaccard_index = True
    continuity = True
    community_events_t0 = True
    community_events_t1 = True


Sample of User Changes Function
*******************************
.. code:: python

    def user_changes_specs(communities: 'list[Community]', nodes: 'list[Nodes]') -> 'list[Nodes]':
        """ returns a list of nodes to delete. it's up to the user which nodes should be killed
        :param communities: list of community objects
        :param nodes: list of node objects
        :return: dead_node_vector: list of nodes to delete (default 10% randomly selected)

Sample of community distribution functions
*******************************************
.. code:: python

    def community_distribution_power_law() -> 'list[int]':
        """ returns a community size distribution in a list

        In this example a power law distribution according to default parameters is returned. User is free to code it's own
        distribution.

        :return: list of community sizes

Sample of node distribution function
************************************
.. code:: python

    def node_distribution_power_law(community_sizes: 'list[int]', retries) -> 'list[int],list[int]':
        """
        returns two node degree distributions: total  and INTRA

        to generate feasible distributions there should not be a skew towards large and small degrees (bathtub)
        maximum degree should be substantially lower than community size???
        :param community_sizes: Community sizes distribution
        :param retries:  retry number if previous sequence non graphic

        :return: lists of total and INTRA node degrees


Parameters for user supplied functions examples
***********************************************
.. parsed-literal::

    community_distribution samples parameters:
	community_distribution_power_law
	desired_number_of_nodes.........................................500
	delta (power exponent)..........................................1.5
	max_community_sizes.............................................300
	min_community_sizes.............................................20

	community_distribution_exponential
	desired_number_of_nodes.........................................500
	beta (scale parameter and mean).................................1
	max_community_sizes.............................................300
	min_community_sizes.............................................20

	community_distribution_random
	desired_number_of_nodes.........................................500
	max_community_sizes.............................................300
	min_community_sizes.............................................20


    node_distribution samples parameters:
	node_distribution_power_law
	mix_ratio (intra to total) .....................................0.7
	fixed (or bernoulli)............................................False
	gamma (power exponent)..........................................2.5
	max_degree......................................................40
	min_degree......................................................8

	node_distribution_exponential
	mix_ratio (intra to total) .....................................0.7
	fixed (or bernoulli)............................................False
	gamma(power exponent)...........................................4
	max_degree......................................................40
	min_degree......................................................8

	node_distribution_random
	pkk (probability of intra link).................................0.2
	pkn (probability of inter link).................................0.002
	fixed (or bernoulli)............................................False
	mix_ratio (intra to total)......................................0.7

