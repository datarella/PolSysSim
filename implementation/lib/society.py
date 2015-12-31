from actor import Actor
from toolbox import *
from input_model import *
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import random
from scipy import linalg
import copy
import igraph as ig
from collections import defaultdict


class Society():
    def __init__(self,group_dict, topic_dict, param_dict, socialism_dict):
        """
        :param group_dict: {groupId:size}, specify how many people in which group
        :param topic_dict: {topicNum: number of policies in that topic} topic_dict 
         provides information about shape of Belief matrix (not necessarily rectangle)
        :param param_dict: {power distribution parameter: a
                            belief vector rotate angle: alpha
                            motivation oscillation amplitude: belta,
                            #candidates in each group: numCandidate,
                            #delegators in each group: numDelegator}
        :param socialism_dict: {'vote_func': vote function}
        """
        self.group_dict = group_dict
        self.topic_dict = topic_dict
        self.param_dict = param_dict
        self.socialism_dict = socialism_dict
        # list holding actors of the society
        self.actors = []
        # sum of actors in all groups
        self.numOfActors = sum([self.group_dict[key] for key in self.group_dict.keys()])
        self.numOfPolicies = sum([self.topic_dict[key] for key in self.topic_dict.keys()])
        self.system_ballot_history = []
        # Tracking simulation time
        self.nSteps = 0
        self.nElections = 0
        self.nBallots = 0
        # Various socialism
        self.vote_func = socialism_dict['vote_func']
        # Voting of last election
        self.votes = None

        self._allocate_actors()
    
    @property
    def BeliefNetwork(self):
        """Compute pairwise belief proximity(reprocical distance) matrix, the 
        ith diagonal of the matrix expose the elligibility of the actor i
        """
        # Belief Network (adjacent matrix, implemented in numpy matrix)
        BN = np.zeros((self.numOfActors, self.numOfActors))
        for a1 in self.actors:
            for a2 in self.actors:
                if a1.groupId is not a2.groupId:
                    BN[a1.idf, a2.idf] = 0
                elif a1.idf == a2.idf:
                    BN[a1.idf, a2.idf] = a1.isEll
                elif a1.idf is not a2.idf:
                    BN[a1.idf, a2.idf] = sigmoid(compute_belief_distance(a1.belief, a2.belief), self.param_dict['sigmoid_decay'])

        return BN

    @property
    def BallotHistoryNetwork(self):
        """Assume Ballot is conducted(if not the matrix will be diagonal), every actor
        has a "vote" on the topic(even it is not a Delegator). The vote of a Delegator
        cause an effect in ballot, the votes of others counts only in the ballot_history 
        of each actor. The similarity of ballot_history is computed and taken as entry in 
        adjacent matrix. Diagonal records the elligibility of actor has nothing to do 
        with ballot history similarity
        """
        BHN = np.zeros((self.numOfActors, self.numOfActors))
        for a1 in self.actors:
            for a2 in self.actors:
                if a1.groupId is not a2.groupId:
                    BHN[a1.idf, a2.idf] = 0
                elif a1.idf == a2.idf:
                    BHN[a1.idf, a2.idf] = a1.isEll
                elif a1.idf is not a2.idf:
                    try:
                        BHN[a1.idf, a2.idf] = sigmoid(compute_ballotHistory_distance(a1.ballot_history, 
                                                        a2.ballot_history), self.param_dict['sigmoid_decay'])
                    except ZeroDivisionError:
                        BHN[a1.idf, a2.idf] = 1 # maximal value for proximity

        return BHN
    
    @property
    def society_partition(self):
        """
        :return: {groupId:list of actors in this partition}
        """
        society_partition = {}
        for groupId in self.group_dict.keys():
            society_partition[groupId]=[]
        for actor in self.actors:
            society_partition[actor.groupId].append(actor.idf)
        return society_partition
    
    @property
    def delegators(self):
        return [actor.idf for actor in self.actors if actor.isDel != 0]
    
    @property
    def population_motivations(self):
        """ Population Motivation
        cross-sectional population statistics (pandas)
        """
        motivations = [actor.motivation for actor in self.actors]
        return pd.Series(motivations)
    
    @property
    def population_beliefs(self):
        """ Population Belief
        """
        topics_dict = {}
        # loop number of topics
        for i in range(len(self.topic_dict)):
            topic = []
            for actor in self.actors:
                topic.append(actor.belief[i])
            topics_dict[i] = pd.DataFrame(np.asarray(topic))
        return pd.Panel(topics_dict)
    
    @property
    def population_beliefs_with_vote(self):
        """Calculate the index of rank 1 component of topic vector,
        and attached to the beliefs panel
        """
        beliefs = self.population_beliefs
        beliefs = beliefs.transpose(2,0,1)
        beliefs['max'] = beliefs.apply(lambda x:x.idxmax(),axis=0)
        beliefs = beliefs.transpose(1,2,0)
        return beliefs
    
    def get_actor_motivation_history(self,i):
        return pd.Series(self.actors[i].motivation_history)
    
    def get_actor_belief_history(self,i):
        belief_stamps = self.actors[i].belief_history
        topics_dict = {}
        # loop number of topics
        for i in range(len(self.topic_dict)):
            topic = []
            for stamp in belief_stamps:
                topic.append(stamp[i])
            topics_dict[i] = pd.DataFrame(np.asarray(topic))
        return pd.Panel(topics_dict)

    
    
    #############################################################################
    # Society Behaviour
    #############################################################################
    
    def _allocate_actors(self):
        # generate random numbers for initializing motivation and belief of actor
        random_motivations = generate_random_motivation(self.param_dict['a'], self.numOfActors)
        random_beliefs = generate_random_beliefs(self.numOfActors, self.numOfPolicies, self.topic_dict)
        count = 0
        
        for groupId,size in self.group_dict.items():
            for i in range(size):
                # build motivation
                m = random_motivations[count]
                # build belief vector
                b = random_beliefs[count]
                self.actors.append(Actor(groupId,m,b, count))
                count+=1
            
    def one_step(self):
        """ Regular change
        actor motivation, belief
        """
        for actor in self.actors:
            actor.modify_motivation(0)
            actor.modify_belief(self.param_dict['alpha'])
            
        self.nSteps += 1
    
    def simul(self, nsteps):
        for step in range(nsteps):
            self.one_step()
            # election
            if self.nSteps%self.param_dict['electionPeriod'] == 0:
                self.election()
            # ballot
            if self.nSteps%self.param_dict['ballotPeriod'] == 0:
                self.ballot()
    
    def _clear_candidates(self):
        """ Set isEll in each actor to be 0 
        """
        for actor in self.actors:
            actor.candidate_history.append([actor.isEll])
            actor.isEll = 0
    
    def _clear_delegators(self):
        """ Set isDel in each actor to be 0
        """
        for actor in self.actors:
            actor.delegation_history.append([actor.isDel])
            actor.isDel = 0
        
    
    def _assign_candidates(self):
        """Choose the highest motivated actor(numCandidate) in
        each partition
        """
        for groupId in self.society_partition.keys():
            constituency = self.society_partition[groupId]
            constituency.sort(key=lambda actor_idf:self.actors[actor_idf].motivation, reverse=True)
            # honor the first numCandidate in the list as candidate
            for i in range(self.param_dict['numCandidate']):
                self.actors[constituency[i]].isEll = 1
    

    def _count_votes(self, votes):
        def _find_leafs():
            froms = set(votes_dict.keys())
            tos = set([votes_dict[f][0] for f in froms])
            leafs = froms - tos
            return leafs
            
        def _isRoot(vote_to):
            return True if votes_dict.get(vote_to) is None else False
        

        # initialise candidate by self-loop in votes
        candidate_votes = {vote_from:1 for vote_from, vote_to in votes if vote_from == vote_to}
        
        # change data structure, remove self-loop
        votes_dict = {vote_from: (vote_to, 1) for vote_from, vote_to in votes if vote_from != vote_to}
        
        
        while votes_dict:
            leafs = _find_leafs()
            for leaf in leafs:
                vote_to, multiplicity = votes_dict[leaf]
                if _isRoot(vote_to):
                        candidate_votes[vote_to] = candidate_votes[vote_to] + multiplicity
                else:
                    votes_dict[vote_to] = (votes_dict[vote_to][0], votes_dict[vote_to][1] + multiplicity)
                votes_dict.pop(leaf)
        # print candidate_votes
                
        return candidate_votes
    
                
    
    def _assign_delegator(self, votes):
        candidate_votes = self._count_votes(votes)
        # print candidate_votes
        delegators = []
        for partition in self.society_partition.values():
            partition_candidate_votes = [(candidate,candidate_votes[candidate]) for candidate 
                                         in candidate_votes.keys() if candidate in partition]
            winner = sorted(partition_candidate_votes, key=lambda x:x[1], reverse=True)[:self.param_dict['numDelegator']]
            delegators.extend(winner)

        # assign delegation to actors
        for (candidate, value)  in delegators:
            self.actors[candidate].isDel = value
        return delegators
    
    
    def election(self):
        """ 
        Use BeliefNetwork(if ballotHistory is empty), otherwise election take the feedback
        from ballot by using BallotHistoryNetwork (or a mixture of BeliefNetwork and 
        BallotHistoryNetwork)
        """
        print '---Election---'
        self.nElections += 1
        
        #clear the candidate from last election
        self._clear_candidates()
        self._clear_delegators()
        self._assign_candidates()
        # we need one proximity matrix for election, setup that
        if self.system_ballot_history == []:
            m = self.BeliefNetwork
        else:
            m = self.BallotHistoryNetwork
        # vote
        votes = self.vote_func(self, m)
        self.votes = votes
        delegators =  self._assign_delegator(votes)
        return delegators, votes
        
    
    def ballot(self):
        print '---Ballot---'
        self.nBallots += 1
        
        def _count_ballot(ballot_df):
            ballot_result={}
            for index,opinion_topics in ballot_df.iterrows():
                weight = self.actors[index].isDel
                for topic_id, opinion_topic in enumerate(opinion_topics):
                    x = (topic_id,opinion_topic)
                    ballot_result[x] = (ballot_result.get(x) if ballot_result.get(x) else 0)+1*weight

            df = [(row, col, ballot_result[(row,col)]) for row,col in ballot_result.keys()]
            ballot_result_df = pd.DataFrame(df).pivot(index=0, columns=1, values=2)
            res = ballot_result_df.idxmax(axis=1).values.tolist()
            return res
            
        
        ballot = []
        # every actor show attitude to policy
        for actor in self.actors:
            actor_opinion =self.population_beliefs_with_vote[:,actor.idf,:].loc['max',:].values.tolist()
            self.actors[actor.idf].ballot_history.append(actor_opinion)
        # decision of delegators has input on the result of ballots
        if self.delegators == []:
            print "Please hold the election first, before ballot"
            
        for delegator in self.delegators:
            delegator_opinioin = self.actors[delegator].ballot_history[-1]
            ballot.append((delegator,delegator_opinioin))
        
        # ballot_result
        index = zip(*ballot)[0]
        delegators_opinion = np.array(zip(*ballot)[1])
        ballot_df = pd.DataFrame(delegators_opinion, index=index)
        # record in system history
        ballot_result = _count_ballot(ballot_df)
        self.system_ballot_history.append(ballot_result)
        
        return ballot_result
