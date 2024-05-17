import abc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from recStrategy import RecStrategy
import pandas as pd

class PreferredArtistStrategy(RecStrategy):
    """
    Handles recommendations for users who have enough information available to properly form recommendations.
    Currently there are two methods for doing this.

    The first is broad recommendations, which averages out the values of each item in the query, and determinies which items to recommend based on it.
    The second is focused recommendations, which only pulls items that are in the same cluster as those in the query.

    **kwargs is used as the parameters for now until the data required is determined.
    """

    def recommend_artist(self, **kwargs):
        return self._get_recommendations_hybrid(**kwargs)

    def recommend_event(self, **kwargs):
        return self._get_recommendations_hybrid(**kwargs)

    def _get_recommendations_broad(self, **kwargs):
        """
        Identifies the most relevant item based on the values of all items in the query.
        """
        # Access the vectors directly using provided indices
        query_vectors = kwargs["vector"][kwargs["query"]]

        # Compute the average vector of the queries
        avg_vector = np.mean(query_vectors, axis=0).reshape(1, -1)

        # Compute cosine similarity between the average query vector and all vectors
        similarities = cosine_similarity(np.asarray(avg_vector), kwargs["vector"])[0]
        most_relevant = np.argsort(-similarities)[:kwargs["num"]]

        # Exclude the original query indices from the recommendations
        recommended_indices = [i for i in most_relevant if i not in kwargs["query"]]

        # Return the corresponding data entries for these indices
        return kwargs["data"].iloc[recommended_indices]
    
    def _get_recommendations_focused(self, **kwargs):
        """
        Identifies the clusters used in the query, and only recommends items based upon it.
        """
        # Use the indices to predict the clusters for the selected rows
        target_clusters = kwargs["model"].predict(kwargs["vector"][kwargs["query"]])
        unique_clusters = set(target_clusters)

        # Find indices of all items in these clusters
        indices_in_clusters = [i for i in range(len(kwargs["data"])) if kwargs["model"].labels_[i] in unique_clusters]

        # Exclude the original query indices and return the data entries
        recommendations_indices = [i for i in indices_in_clusters if i not in kwargs["query"]]
        if len(recommendations_indices) > kwargs["num"]:
            return kwargs["data"].iloc[recommendations_indices[:15]]
        return kwargs["data"].iloc[recommendations_indices]
    
    def _get_recommendations_hybrid(self, **kwargs):
        """
        Provides a mix of broad and focused, for a more relevant reccommendation
        """
        # First, get a preliminary list of recommendations based on clustering
        cluster_recommendations = self._get_recommendations_focused(**kwargs)

        # Check if the cluster recommendations returned enough results
        if len(cluster_recommendations) < kwargs["num"]:
            # If not enough recommendations, broaden the search or fill in with broad recommendations
            broad_recommendations = self._get_recommendations_broad(**kwargs)
            combined_recommendations = pd.concat([cluster_recommendations, broad_recommendations]).drop_duplicates()
            return combined_recommendations.head(kwargs["num"])

        # Return the refined list based on similarity
        return cluster_recommendations.head(kwargs["num"])