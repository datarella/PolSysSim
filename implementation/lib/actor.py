import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import random
from scipy import linalg as LA
import copy
import igraph as ig
from collections import defaultdict

class Actor():
    def __init__(self, groupId, motivation, belief, _id, topic_relevance):
        """
        param motivation: single numbe (0,1]
        param belief: array of "vectors" (topic vector)
        param _id: identification of actor(surrogate key)
        param topic_relevance: relevance of a topic to this actor
        """
        self._id = _id
        self.motivation = motivation
        # explicit political beliefs
        # (relevant) belief matrix
        self.belief = belief
        self.groupId = groupId
        self.topic_relevance = topic_relevance
        
        #self.relevant_belief = self._compute_relevant_belief()
        
    @property
    def relevant_belief(self):
        return {(topic_id, policy_id): self.belief[(topic_id, policy_id)]*self.topic_relevance[topic_id]  for topic_id, policy_id in self.belief.keys()}
    
    # pd.DataFrame with (topic_id,policy_id) as index
    @property
    def policy_rank(self):
        df = pd.DataFrame(self.relevant_belief.values(), 
                          index=pd.MultiIndex.from_tuples(self.relevant_belief.keys(), names=["topic_id", "policy_id"]),
                          columns=["value"]).sort_index(level=0)
        
        df["rank"] = df.groupby(level=["topic_id"])["value"].rank(ascending=False)
        return df
        
        
        """
        self.isDel = 0
        self.isEli = 0
        # Keep trace of the actor history
        self.motivation_history = []
        self.belief_history = []
        self.delegation_history = []
        self.candidate_history = []
        
        #Initialization
        self.motivation_history.append(self.motivation)
        self.belief_history.append(self.belief)
        self.delegation_history.append(self.isDel)
        self.candidate_history.append(self.isEli)
        # implicit political beliefs,each entry [vote_for_policy_topic1, vote_for_policy_topic2,...]
        self.ballot_history = []
        """
        
        """
        def update_motivation_each_step(self, belta):
        # Update Motivation at each simulation step
        
        if self.motivation+belta < Actor.MOTIVATION_UPPER_BOUNDARY and self.motivation+belta > Actor.MOTIVATION_LOWER_BOUNDARY:
            self.motivation += belta
            
        motivation_copy = copy.deepcopy(self.motivation)
        """

        """
        def modify_belief(self, alpha):
        "" Update Belief at each simulation step
        
        Rotate each topic vector, if the number
        of alternatives is larger than 2 only the first two
        dimensition is rotated (todo: change this to rotate
        random two dimension of the vector)
        
        alpha is the CONST part of random change parameter [-\pi/100, \pi/100]
        i.g. pi/100
        ""
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
        """
