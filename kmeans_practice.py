import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

#performance_data.drop(columns=['UUID'])
# user_data.head()
# user_data.describe()

scaler = StandardScaler()
user_data[['latitude_T', 'longitude_T']] = scaler.fit_transform(user_data[['latitude', 'longitude']])
# user_data

performance_data[['latitude_T', 'longitude_T']] = scaler.fit_transform(performance_data[['latitude', 'longitude']])
performance_data

performance_kmeans = KMeans(n_clusters=3, n_init=10)
performance_kmeans.fit(performance_data[['latitude_T', 'longitude_T']])

user_kmeans = KMeans(n_clusters=2, n_init=10)
user_kmeans.fit(user_data[['latitude_T', 'longitude_T']])

performance_data['kmenas_3'] = performance_kmeans.labels_
user_data['kmenas_3'] = user_kmeans.labels_
performance_data

plt.scatter(x=performance_data['latitude_T'], y=performance_data['longitude_T'], c=performance_data['kmenas_3'])
# plt.xlim(39,42)
# plt.ylim(-75, -88)
plt.xlim(-3,5)
plt.ylim(4, -3)
plt.show()