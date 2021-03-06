import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import igraph as ig

class SocietyVisualizer():
    def __init__(self, society):
        self.society = society
    
    def visualize_population_motivation(self, kind='hist'):
        """ Plot the histogram of motivations of actors in the society
        cross-sectional study
        :param kind: what kind of plot is needed on motivation
        """
        population_motivations = self.society.population_motivations
        fig, ax = pl.subplots()
        #fig.set_size_inches(18.5,10.5)
        pd.Series.plot(population_motivations,kind=kind, ax = ax)
        pl.show()
        
    def visualize_population_beliefs(self):
        beliefs = self.society.population_beliefs_with_vote
        nTopics = beliefs.shape[0]
        fig,axes = pl.subplots(nrows=nTopics)
        #fig.set_size_inches(18.5,10.5)
        for i in range(nTopics):
            value_counts = beliefs[i]['max'].value_counts()#.plot(kind='hist',ax=axes[i])
            value_counts = zip(*[(value,count) for value,count in value_counts.iteritems()])
            values,counts = value_counts[0],value_counts[1]
            axes[i].pie(counts, labels=values,autopct='%1.1f%%',shadow=True, startangle=90)
            axes[i].set_title('Topic %s'%(i,))
        pl.show()
    
    def visualize_actor_motivation_longitudinal(self,i):
        motivations = self.society.get_actor_motivation_history(i)
        fig, ax = pl.subplots()
        fig.set_size_inches(18.5,10.5)
        pd.Series.plot(motivations, ax = ax)
        pl.show()
    
    def visualize_actor_belief_longitudinal(self, i):
        belief_history = self.society.get_actor_belief_history(i)
        numTopics = len(belief_history)
        fig, axes = pl.subplots(nrows=numTopics)
        fig.set_size_inches(18.5,10.5)
        for i in range(numTopics):
            pd.DataFrame.plot(belief_history[i], ax=axes[i])
        pl.show()
    
    def visualize_Network_heatmap(self, network):
        fig, ax = pl.subplots()
        fig.set_size_inches(18.5,10.5)
        pl.imshow(network)
        pl.colorbar(orientation='vertical')
        pl.show()
