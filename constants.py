"""
module with names that should not be changed during the execution of the system.

Some are constants (INTRA, INTER) others control system behaviour

"""

# NO CHANGES
# to index array in node object: ugly but I found no other way in python
INTRA = 0
INTER = 1

# in case we want the max degree of a node in a community
# < community_size *( max degree in the largest community / largest community size)
# set ADJUSTED_MAX_DEGREE = True
ADJUSTED_MAX_DEGREE = False

# maximum number of cycles to try to link a community or a network before giving up
MAX_CYCLES = 100

# do not allow disjoint communities (can happen if minimum degree < community size -1
ALLOW_DISJOINT_COMMUNITIES = False

ZERO_THRESHOLD = 1.e-14         # Threshold for zero eigenvalues

# joint distribution of node degrees, auxiliary variables
ALFA_MAX = 5           # Strongest dissortative (used for beta_assortativity distribution as alfa_assortativity
# and beta_assortativity parameters)
BETA_MAX = 5           # Strongest assortative

# Create a global timestamp for dynamic time management
class Ts:
    timestamp = 0


Ts()

