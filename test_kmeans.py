from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import numpy as np

# Convert Excel sheets to CSV files
excel_file_path = "sample-data.xlsx"

excel_data = pd.read_excel(excel_file_path, sheet_name=None, engine="openpyxl")

for sheet_name, df in excel_data.items():
    csv_file_path = f"{sheet_name}.csv"
    df.to_csv(csv_file_path, index=False)
    # print(f"Sheet '{sheet_name}' has been converted to CSV file '{csv_file_path}'.")

# Read the data
artist = pd.read_csv(
    "artistdata.csv",
    usecols=["drag_name", "description", "location", "reality_tv", "season"],
    dtype={
        "drag_name": "str",
        "description": "str",
        "location": "str",
        "reality_tv": "str",
        "season": "str",
    },
)


# Encode categorical variables
label_encoder = LabelEncoder()
artist["location_code"] = label_encoder.fit_transform(artist["location"])
# artist["reality_tv_code"] = label_encoder.fit_transform(artist["reality_tv"])
artist["season_code"] = label_encoder.fit_transform(artist["season"])

# Select features
features = ["location_code", "season_code"]

# print the season code
print(artist["season_code"])

# # # Calculate clustering metrics for different values of k
silhouette_scores = []
for k in range(2, 12):  # Try different values of k
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(artist[features])
    silhouette_scores.append(silhouette_score(artist[features], kmeans.labels_))

# Find the optimal k value based on silhouette score
optimal_k = np.argmax(silhouette_scores) + 2  # Add 2 because we started from k=2
print("Optimal number of clusters:", optimal_k)


# Fit KMeans with the optimal k value
optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
optimal_kmeans.fit(artist[features])

# Predict cluster labels for each data point
cluster_labels = optimal_kmeans.labels_

# Add cluster labels to the DataFrame
artist["cluster"] = cluster_labels

# Print data points for each cluster
for cluster_id in range(optimal_k):
    print(f"Cluster {cluster_id}:")
    print(artist[artist["cluster"] == cluster_id])
    print("\n")

# # Plot silhouette scores
plt.plot(range(2, 12), silhouette_scores, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Values of k")
plt.xticks(range(2, 12))
plt.grid(True)
plt.show()

# # Generate some sample data
# np.random.seed(0)
# X = np.random.randn(100, 2)  # 100 data points in 2 dimensions

# # Define the number of clusters
# k = 3

# # Initialize the KMeans model
# kmeans = KMeans(n_clusters=k)

# # Fit the model to the data
# kmeans.fit(X)

# # Get the cluster centroids and labels
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# # Plot the data points and cluster centroids
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("K-means Clustering")
# plt.show()
