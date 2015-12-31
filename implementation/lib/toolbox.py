import numpy as np
import scipy.linalg as LA

def norm2unit(v):
    """ Normalize a vector to unit vector
    :param v: 1D numpy array
    
    >>> norm2unit(np.asarray([0.1,0.4,-1.2]))
    [ 0.07881104  0.31524416 -0.94573249]
    >>> norm2unit(np.asarray([0.0,0.0]))
    [0,0,0,0]
    """
    magnitude = np.sqrt(np.sum(v**2))
    if magnitude == 0:
        return v
    return v/magnitude



def compute_belief_distance(b1,b2):
    # belief1, belief2 should be the same length
    assert(len(b1)==len(b2))
    length = len(b1)
    return sum([LA.norm(b1[i]-b2[i]) for i in range(length)])

def compute_ballot_distance(ba1, ba2):
    """
    :param ba1: array, each array contains 
                Policy id of the voted of topic
    if position do not match +1, if match +0
    then summarize to get the distance of vote
    between two actors in one-round ballot
    """
    s = 0
    assert(len(ba1)==len(ba2))
    n = len(ba1)
    for i in range(n):
        if ba1[i] != ba2[i]:
            s +=1
    return s


def compute_ballotHistory_distance(bh1, bh2):
    """
    ballot histoy is a list of array, first piecewise
    ballot distance is computed (using compute_ballot_distane)
    then aggregate and normalize along the list
    """
    # import pdb; pdb.set_trace()
    assert(len(bh1)==len(bh2))
    n = len(bh1)
    assert(n >0 )
    dist = sum([(compute_ballot_distance(bh1[i],bh2[i]))**2 for i in range(n)])
    # normalize
    dist = dist/float(n)
    return dist

def get_candidate_mask_matrix(society):
    """Column Oriented

    mask[:,i] = [1,1,....] means actor i is a candidate for election
    reversely, mask[:,j] = [0,0,.....] means actor j is not a candidate 
    """
    mask = np.zeros((society.numOfActors, society.numOfActors))
    for actor in society.actors:
        if actor.isEll != 0:   #candidate
            mask[:, actor.idf] = 1

    # in representative system, actor do not vote for candidate in 
    # different constituency
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            # two candidates do not in the same group
            if society.actors[row].groupId != society.actors[col].groupId:
                mask[row,col] = 0


    return mask


def get_motivation_mask_matrix(society):
        """Row Oriented
        
        mask[i,j]==1 True means ai.motivation < aj.motivation
        otherwise equals 0
        diagonal is always 1
        
        Delegation flows from low motivation to high motivation actor
        """
        
        mask = np.zeros((society.numOfActors, society.numOfActors))
        for a1 in society.actors:
            for a2 in society.actors:
                if a1.groupId is not a2.groupId:
                    mask[a1.idf, a2.idf] = 0
                    continue
                elif a1.motivation <= a2.motivation:
                    mask[a1.idf, a2.idf] =1

        return mask

def sigmoid(z, decay):
    """
    :param decay: should be negative number
    """
    s = 1.0 / (1.0 + np.exp(-decay * z))
    return s

"""
Visualize DelegationNetwork

g = ig.Graph.TupleList(s.society.votes, directed=True)
layout = g.layout('kk')
ig.plot(g, layout = layout)
"""


if __name__ == '__main__':
    r = compute_ballotHistory_distance([[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    print r