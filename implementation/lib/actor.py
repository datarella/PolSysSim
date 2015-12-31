import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import random
from scipy import linalg as LA
import copy
import igraph as ig
from collections import defaultdict

class Actor():
    MOTIVATION_UPPER_BOUNDARY = 1.0
    MOTIVATION_LOWER_BOUNDARY = 0.0
    def __init__(self, groupId, motivation, belief, idf):
        """
        param motivation: single numbe (0,1]
        param belief: array of "vectors" (topic vector)
        param idf: identification of actor, its generation sequence number
        """
        self.idf = idf
        self.motivation = motivation
        # explicit political beliefs
        self.belief = belief
        self.groupId = groupId
        self.isDel = 0
        self.isEll = 0
        #---------- Actor History-------------
        self.motivation_history = []
        self.belief_history = []
        self.delegation_history = []
        self.candidate_history = []
        self.motivation_history.append(self.motivation)
        self.belief_history.append(self.belief)
        self.delegation_history.append(self.isDel)
        self.candidate_history.append(self.isEll)
        # implicit political beliefs,each entry [vote_for_policy_topic1, vote_for_policy_topic2,...]
        self.ballot_history = []

    def modify_motivation(self, belta):
        """ Update Motivation
        """
        if self.motivation+belta < Actor.MOTIVATION_UPPER_BOUNDARY and self.motivation+belta > Actor.MOTIVATION_LOWER_BOUNDARY:
            self.motivation += belta
            
        motivation_copy = copy.deepcopy(self.motivation)
        self.motivation_history.append(motivation_copy)


    def modify_belief(self, alpha):
        """ Rotate each topic vector, if the number
        of alternatives is larger than 2 only the first two
        dimensition is rotated (todo: change this to rotate
        random two dimension of the vector)
        
        alpha is the CONST part of random change parameter [-\pi/100, \pi/100]
        i.g. pi/100
        """
        for i in range(len(self.belief)):
            numAlters = len(self.belief[i])
            #construct matrix
            if numAlters == 2:
                delta = random.uniform(-alpha, alpha)
                cos = np.cos(delta)
                sin = np.sin(delta)

                D = np.array([[cos,-sin],
                             [sin,cos]])
            elif numAlters == 1:
                delta = random.uniform(-alpha, alpha)
                D = np.cos(delta)
            elif numAlters > 2:
                delta = random.uniform(-alpha, alpha)
                cos = np.cos(delta)
                sin = np.sin(delta)

                d = np.array([[cos,-sin],
                             [sin,cos]])
                ones = [1]*(numAlters-2)
                D = LA.block_diag(d,*ones)
                
            self.belief[i] = np.dot(D,self.belief[i])
        belief_copy = copy.deepcopy(self.belief)
        #belief_copy = belief_copy.tolist()
        # actor keeps track of its belief history
        self.belief_history.append(belief_copy)


if __name__ == '__main__':
    a = Actor(1,2,3,4)
    a.modify_motivation
    a.modify_belief
