import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Read the CSV data
contestants = pd.read_csv(
    'contestant.csv',
    usecols=['id', 'name', 'original_season_id', 'real_name', 'dob', 'gender', 'hometown', 'location'],
    dtype={'id': 'str', 'name': 'str', 'original_season_id': 'str', 'real_name': 'str', 'dob': 'str', 'gender': 'str', 'hometown': 'str', 'location': 'str'}
)

# Print first 5 rows of contestants data
print("Contestants DataFrame:")
print(contestants.head())

# Pivot Table Example: Counting Contestants by Gender and Hometown
contestants_pivot = contestants.pivot_table(
    index='name',
    columns='hometown',
    values='id',
    aggfunc='count',
    fill_value=0
)

print("\nContestants Pivot Table by Gender and Hometown:")
print(contestants_pivot)

# Sparse matrix
contestants_pivot_matrix = csr_matrix(contestants_pivot.values)

# Display matrix
print("\nSparse Matrix:")
print(contestants_pivot_matrix)

# KNN model with cosine similarity metric
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit model using pivot table matrix
model_knn.fit(contestants_pivot_matrix)

# Random choice (within the range of pivot table indices)
query_index = np.random.choice(contestants_pivot.shape[0])

# Print the query index and its corresponding gender
print("\nQuery Index:")
print(query_index)
print(f'Querying for: {contestants_pivot.index[query_index]}')

# Get nearest neighbors (using the pivot table)
distances, indices = model_knn.kneighbors(contestants_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

# Print recommendations
print("\nRecommendations:")
for i in range(0, len(distances.flatten())):
    if i == 0:
        print(f'Recommendations for {contestants_pivot.index[query_index]}:\n')
    else:
        print(f'{i}: {contestants_pivot.index[indices.flatten()[i]]}, with distance of {distances.flatten()[i]:.4f}')
