import sys
sys.path.append('../../')

import pickle as pkl
from panav.hybrid import reduced_agents_HG

def load_room(path_to_base_room,N_agent):
    with open(path_to_base_room,'rb') as fp:
        HG = pkl.load(fp)
        return reduced_agents_HG(HG,N_agent)

def load_env_N(path_to_env,N_agent):
    return load_env(path_to_env+str(N_agent)+'.pkl')

def load_env(path_to_env):
    with open(path_to_env,'rb') as fp:
        return pkl.load(fp)