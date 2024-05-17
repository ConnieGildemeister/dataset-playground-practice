import abc
import numpy as np
from sklearn.cluster import KMeans
from recStrategy import RecStrategy

class ColdStartStrategy(RecStrategy):
    """
    Handles recommendations for users who do not have enough information available to properly form recommendations.
    Pulls artist/events randomly from the known pool, and gives recommendations based on the selection.

    **kwargs is used as the parameters for now until the data required is determined.
    """


    def recommend_artist(self, **kwargs):
        return self.get_recommendations(**kwargs)

    def recommend_event(self, **kwargs):
        return self.get_recommendations(**kwargs)

    def get_recommendations(self, **kwargs):

        #Choose how many items we want to base this on
        num_random = min(5, kwargs["num"])
        indicies = np.random.choice(len(kwargs["data"]), num_random, replace=False)
             
        recommendations = []

        #Predict the clusters from the selected items
        clusters = kwargs["model"].predict(kwargs["vector"][indicies])

        #Iterate through the clusters, adding related items from the cluster until the amount wanted is aquired.
        for cluster in clusters:
            if(len(recommendations) >= kwargs["num"]):
                    break
            cluster_indicies = np.where(kwargs["model"].labels_ == cluster)[0]
            np.random.shuffle(cluster_indicies)
            for i in cluster_indicies[:kwargs["num"]]:
                if(len(recommendations) >= kwargs["num"]):
                    break
                print(len(recommendations) + 1, "/", kwargs["num"])
                recommendations.append(i)
        
        return kwargs["data"].iloc[recommendations]