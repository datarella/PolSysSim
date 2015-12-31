from toolbox import get_motivation_mask_matrix
import numpy as np


def vote_liquid_democracy(society, m):
        """ Voting rules
        * Each candidate has 1 "delegation-coin"
        * Candidates only vote for themselves
        * Each non-candidate actor give vote to (nearst)actors with higher motivation
        * If such actor ((nearst)actors with higher motivation) does not exist, then vote
        directly to candidate

        :param m: proximity matrix
        """

        N = m.shape[0]
        assert(m.shape[0] == m.shape[1])
        # ensure only low motivation actor vote for high motivation actor
        mask = get_motivation_mask_matrix(society)
        masked_m = np.multiply(m, mask)
        
        # collect delegation situation
        votes = []
        for voted_from in range(N): #nrow
            if masked_m[voted_from, voted_from] == 1: 
                votes.append((voted_from, voted_from)) #candidate vote for themselves
            # other situations
            voted_to = np.argmax(masked_m[voted_from])
            assert(masked_m[voted_from, voted_to] > 0)  # proximity do not equal 0
            votes.append((voted_from, voted_to)) 
            
        return votes
        
