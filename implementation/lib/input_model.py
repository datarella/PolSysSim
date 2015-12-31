import numpy as np
from toolbox import *

def generate_random_motivation(a, numOfActors):
    """ Fake up motivations for actor
    """
    return np.random.power(a, size=numOfActors)
    
def generate_random_beliefs(numOfActors, numOfPolicies, topic_dict):
    """ Fake up beliefs for actor
    """
    # 1 dimension : policies, 2 dimension: actors
    random_beliefs = np.asarray([np.random.normal(0,1,numOfActors) for i in range(numOfPolicies)])
    random_beliefs = random_beliefs.T
    # calculate the split point
    split_point = np.cumsum(np.asarray([topic_dict[key] for key in topic_dict.keys()]))
    # the last one holds the rest of splitting which is always empty in out case
    splitted_random_beliefs_list = np.hsplit(random_beliefs, split_point)[:-1]
    random_beliefs_res = []
    for i in range(numOfActors):
        belief_vectors = []
        for j in range(len(splitted_random_beliefs_list)):
            belief_vectors.append(norm2unit(splitted_random_beliefs_list[j][i]))
        random_beliefs_res.append(belief_vectors)
    # random_beliefs_res = np.asarray(random_beliefs_res)
    return random_beliefs_res
