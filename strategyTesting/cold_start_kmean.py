import abc
import numpy as np
from sklearn.cluster import KMeans
from recStrategy import RecStrategy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class ColdStartStrategyAlt(RecStrategy):
    """
    Testing using kmeans without knowing the matrix, currently does not work.
    """


    def recommend_artist(self, **kwargs):
        return self.get_recommendations(**kwargs)

    def recommend_event(self, **kwargs):
        return self.get_recommendations(**kwargs)

    def get_recommendations(self, **kwargs):

        #Choose how many items we want to base this on
        num_random = min(5, kwargs["num"])
        pipeline = make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False))
        data = pipeline.fit_transform(kwargs["data"]["test_data"])
             
        recommendations = []

        #Predict the clusters from the selected items
        clusters = kwargs["model"].predict(data)

        #Iterate through the clusters, adding related items from the cluster until the amount needed is aquired.
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