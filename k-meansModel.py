import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data
artists = {
    "Artist1": {"location": [40.7128, -74.0060], "tv_contestant": True},
    "Artist2": {"location": [34.0522, -118.2437], "tv_contestant": False},
    "Artist3": {"location": [41.8781, -87.6298], "tv_contestant": True},
    "Artist4": {"location": [51.5074, -0.1278], "tv_contestant": False},
    "Artist5": {"location": [48.8566, 2.3522], "tv_contestant": True},
    "Artist6": {"location": [35.6895, 139.6917], "tv_contestant": False},
}


class KMeansModel(KMeans):
    def __init__(self):
        KMeans.__init__(self)
        # Standardize the data
        self.__scaled_data = self.__standardize_data()
        # Find the optimal number of clusters
        self.__num_clusters = self.__find_optimal_k(self.__scaled_data)
        # Perform clustering
        self.__make_clusters()

    def __standardize_data(self):
        # Convert data to numpy array
        data = np.array([[*artist["location"], artist["tv_contestant"]] for artist in artists.values()])
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def __make_clusters(self):
        # Apply K-means clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=self.__num_clusters, random_state=42)
        kmeans.fit(self.__scaled_data)

        # Get cluster labels
        cluster_labels = kmeans.labels_

        # Add cluster labels to the original data
        for artist, cluster_label in zip(artists.keys(), cluster_labels):
            artists[artist]["cluster"] = cluster_label

        # Print the result
        for artist, details in artists.items():
            print(f"{artist}: Cluster {details['cluster']}")

    def __find_optimal_k(self, data, max_k=5):
        # Initialize lists to store inertias
        wcss = []

        # Iterate over each k value
        for k in range(1, max_k + 1):
            # Fit KMeans model with current k value
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            # Append the inertia (within-cluster sum of squares) to the list
            wcss.append(kmeans.inertia_)

        # Plot the Elbow Method (optional)
        plt.plot(range(1, max_k + 1), wcss, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.xticks(range(1, max_k + 1))
        plt.grid(True)
        # plt.show()

        # Find the optimal k where the elbow is located
        # A simple heuristic: optimal k is where the decrease in WCSS slows down
        diff = np.diff(wcss)
        second_diff = np.diff(diff)
        optimal_k = np.argmax(second_diff) + 2  # +1 for first diff, +1 for 0-based index
        return optimal_k


model = KMeansModel()
