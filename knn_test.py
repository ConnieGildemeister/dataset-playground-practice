import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors

user_data = pd.read_excel('/Users/yerimmoon/Documents/3800/userdata.xlsx')
performance_data = pd.read_excel('/Users/yerimmoon/Documents/3800/performancedata.xlsx')

# Reset the index to create a new column 'user_id'
user_data.reset_index(inplace=True)
user_data.rename(columns={'index': 'user_id'}, inplace=True)
user_data['user_id'] = range(1, len(user_data) + 1)
# user_data.head()

geolocator = Nominatim(user_agent="recommender")
user_data['latitude'] = user_data['closest_city'].apply(lambda x: geolocator.geocode(x).latitude)
user_data['longitude'] = user_data['closest_city'].apply(lambda x: geolocator.geocode(x).longitude)
performance_data['latitude'] = performance_data['location'].apply(lambda x: geolocator.geocode(x).latitude)
performance_data['longitude'] = performance_data['location'].apply(lambda x: geolocator.geocode(x).longitude)

k = 5
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

X = performance_data[['latitude', 'longitude']]

knn.fit(X)

# Function to recommend performers to a user
def recommend_performers(user_row):
    # Get user's location
    user_location = [user_row['latitude'], user_row['longitude']]
    
    # Find K nearest performers
    distances, indices = knn.kneighbors([user_location])
    
    # Get recommended performers
    recommended_performers = performance_data.iloc[indices[0]]
    
    return recommended_performers

# Loop through each user and recommend performers
for index, user_row in user_data.iterrows():
    user_id = user_row['user_id']
    recommended_performers = recommend_performers(user_row)
    
    print(f"Recommendations for User {user_id}:")
    print(recommended_performers)
    print("\n")