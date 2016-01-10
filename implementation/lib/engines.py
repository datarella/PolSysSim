import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import mode



def my_mode_func(x):
    new_column = x.ix[:,-1]
    res = new_column.mode()
    if res.empty:
        # pd mode Empty if nothing has 2+ occurrences
        #scipy.stats.mode
        res = mode(new_column)[0]
        return res
    else:
        return res.values


class BallotEngine:

    @staticmethod
    def ballot(topic_ids, actors):
        """
        :param actors: A_(D) set
        """
        res = {}
        for topic_id in topic_ids:
            # initialize ballot_result for this topic_id
            # multiple rounds possible if a tie
            select_round = 1
            ballot_result = pd.DataFrame({"actor_id":[actor._id for actor in actors]}).set_index("actor_id")
            # call ballot_one_topic recursively
            selected_policy_id, ballot_result = BallotEngine.ballot_one_topic(topic_id,
                                                                             actors,
                                                                             select_round,
                                                                             ballot_result)
            res[topic_id] = {"selected_policy_id":selected_policy_id, "ballot_result":ballot_result}       
        return res
            
    @staticmethod
    def ballot_one_topic(topic_id, actors, select_round, ballot_result):
        # collect votes to topic_id
        # first round, rank=1
        
        # import pdb;pdb.set_trace()
        
        ballot_votes = {actor._id: int(pd.pivot_table(actor.policy_rank.reset_index(), index=["topic_id", "rank"], values=["policy_id"]).ix[(topic_id,select_round)]) for actor in actors}
        
        ballot_result = pd.concat([ballot_result, pd.DataFrame({"actor_id": ballot_votes.keys(), "round %s" %(select_round,):ballot_votes.values()}).set_index("actor_id")], axis=1)


        selected_policy_id_list = list(my_mode_func(ballot_result))
        
        # check if run-off
        if len(selected_policy_id_list)>1:
            # remain set of actor_id
            remain_actors_id = set(ballot_result[~ballot_result["round %s" %(select_round,)].isin(selected_policy_id_list)].index.values)
            select_round += 1
            remain_actors = [actor for actor in actors if actor._id in remain_actors_id]
            
            return BallotEngine.ballot_one_topic(topic_id, remain_actors, select_round, ballot_result)
            
        else:
            return selected_policy_id_list, ballot_result