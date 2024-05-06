import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Read data
movies_df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Recommendation_complete_tutorial/master/KNN%20Movie%20Recommendation/movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
rating_df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Recommendation_complete_tutorial/master/KNN%20Movie%20Recommendation/ratings.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# Print first 5 rows of movies and ratings
print("Movies DataFrame:")
print(movies_df.head())
print("\nRatings DataFrame:")
print(rating_df.head())

# Merge datasets
df = pd.merge(rating_df, movies_df, on='movieId')

# Drop missing values (if title is NaN then will remove the values)
combine_movie_rating = df.dropna(axis=0, subset=['title'])

# Group by title and rating
movie_ratingCount = combine_movie_rating.groupby(by=['title'])['rating'].count().reset_index().rename(columns={'rating': 'totalRatingCount'})[['title', 'totalRatingCount']]

# Merge datasets
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title', how='left')

# Print first 5 rows
print("\nMerged DataFrame:")
print(rating_with_totalRatingCount.head())

# Format
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Summary statistics
print("\nSummary Statistics:")
print(movie_ratingCount['totalRatingCount'].describe())

# Threshold
popularity_threshold = 50

# Query
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

# Print first 5 rows and shape of the data
print("\nPopular Movies:")
print(rating_popular_movie.head())
print("\nShape of the Data:")
print(rating_popular_movie.shape)

# Pivot table
movie_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)

# Print first 5 rows
print("\nPivot Table:")
print(movie_features_df.head())

# Sparse matrix
movie_features_df_matrix = csr_matrix(movie_features_df.values)

# Display matrix
print("\nSparse Matrix:")
print(movie_features_df_matrix)

# KNN model with cosine similarity metric
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit model
model_knn.fit(movie_features_df_matrix)

# Random choice
query_index = np.random.choice(movie_features_df.shape[0])

# Print index
print("\nQuery Index:")
print(query_index)

# Get nearest neighbors
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

# Print recommendations
print("\nRecommendations:")
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))

# Countplot
# sns.countplot(x='rating', data=rating_df, palette='deep')
# plt.savefig('countplot.png')
# plt.show()
