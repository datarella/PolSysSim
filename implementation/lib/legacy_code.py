"""    
    def create_DelegationNetwork(self, votes):
        "Adjacent Matrix
        :param votes: Result of voting, list of tuple
        "
        DelegationNetwork = np.zeros((self.numOfActors, self.numOfActors))
        for vote_from, vote_to in votes:
            DelegationNetwork[vote_from, vote_to] = 1
        return DelegationNetwork
    
    def _linkout(self, DN, i):
        "
        :param i: the ith actor
        :return j: the jth actor(outlinker), if not linked out then return None
        "
        row = DN[i]
        s = sum(DN[i])
        # assert(s<=1)
        isLinkout = True if s==1 else False
        if not isLinkout:
            return None
        j = np.nonzero(row)
        return j
        
    
    def _count_votes(self, DelegationNetwork):
        candidates = []
        DN = copy.deepcopy(DelegationNetwork)
        assert(DN.shape[0]==DN.shape[1])
        n = DN.shape[0]

        for i in range(n):
            col = DN[:,i]
            votes_to_i = sum(col)
            # check if there is link out
            j = self._linkout(DN,i)
            if j is not None:
                # transfer the vote_to_i to j
                DN[i,j] += votes_to_i
                DN[:,i] = 0
            else:
                candidates.append((i,votes_to_i))
            
        print count
        return candidates
"""
