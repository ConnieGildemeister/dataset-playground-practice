from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

import pandas as pd

#Uses the additional sample data the client provided, change dir if needed
events = pd.read_csv("additional-data.csv", usecols=["event_date","event_name","venue_name","full_address","location","event_info","base_price"])
events["test_data"] = events["event_name"] + " " + events["event_info"]
print(events)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(events["test_data"])

#Scalar
pipeline = make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False))
data = pipeline.fit_transform(events["test_data"])
print("----------Scalar Data----------\n",data)

# Apply KMeans with k=3
k = 27
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)

print("\n----------Cold Start----------\n")

print("\n----------Cold Start----------\n")
from cold_start_strategy import ColdStartStrategy
from preferred_artist_strategy import PreferredArtistStrategy
from cold_start_kmean import ColdStartStrategyAlt

cs = ColdStartStrategy()
pa = PreferredArtistStrategy()
csa = ColdStartStrategyAlt()


##print(cs.recommend_event(num=15, model=kmeans,vector=X, data=descriptions))

queries = []
query = ["Brunch is a Drag at The Craftsman - Wicked & Wizard of Oz", "Tavern & Tiaras - 8/23/24"]

for item in query:
    index = events.index[events['event_name'] == item].tolist()
    queries.append(index[0])

print("Queries:", queries)
print("Vector:\n", X)
print("----------Results----------")
#print(cs.recommend_event(num=15, model=kmeans,vector=data, data=events))
recommendations = csa.recommend_event(num=15, model=kmeans, data=events)
#recommendations = pa.recommend_artist(query=queries,num=5, model=kmeans,vector=data, data=events)
print(recommendations)