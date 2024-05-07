import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# Convert Excel sheets to CSV files
excel_file_path = "sample-data.xlsx"

excel_data = pd.read_excel(excel_file_path, sheet_name=None, engine="openpyxl")

for sheet_name, df in excel_data.items():
    csv_file_path = f"{sheet_name}.csv"
    df.to_csv(csv_file_path, index=False)
    # print(f"Sheet '{sheet_name}' has been converted to CSV file '{csv_file_path}'.")

# Read the 'userdata.csv' file
users = pd.read_csv(
    "userdata.csv",
    usecols=["email", "first_name", "last_name", "dob", "closest_city"],
    dtype={
        "email": "str",
        "first_name": "str",
        "last_name": "str",
        "dob": "str",
        "closest_city": "str",
    },
)

# reading performance data
performances = pd.read_csv(
    "performancedata.csv",
    usecols=["event_name", "date", "location", "artist", "age_minimum"],
    dtype={
        "event_name": "str",
        "date": "str",
        "location": "str",
        "artist": "str",
        "age_minimum": "str",
    },
)

# Parse 'dob' column to datetime format
users["dob"] = pd.to_datetime(users["dob"])
performances["date"] = pd.to_datetime(performances["date"])

# Rename the "location" column to "closest_city" in the performances DataFrame
performances.rename(columns={"location": "closest_city"}, inplace=True)


# Function to calculate age
def calculate_age(dob):
    current_date = datetime.now()
    age = (
        current_date.year
        - dob.year
        - ((current_date.month, current_date.day) < (dob.month, dob.day))
    )
    return age


# Apply the function to each row
users["age"] = users["dob"].apply(calculate_age)

# Merge 'first_name' and 'last_name' into 'name'
first_name = "first_name"
last_name = "last_name"
name = [first_name, last_name]

users["name"] = users[first_name].astype(str) + " " + users[last_name].astype(str)

# Drop 'first_name' and 'last_name' columns
users_clean = users.drop(columns=name)
users_clean = users_clean.drop(columns="dob")

# Reorder columns with 'name' first
new_column_order = ["name"] + [col for col in users_clean.columns if col != "name"]
users = users_clean[new_column_order]

# Merge users and performances on closest_city
merged_data = pd.merge(users, performances, how="inner", on="closest_city")
merged_data["merged_id"] = range(1, len(merged_data) + 1)

# Rearrange the columns so that the "merge_id" column comes first
cols = merged_data.columns.tolist()
cols = ["merged_id"] + [col for col in cols if col != "merged_id"]
merged_data = merged_data[cols]

# print(merged_data.head())

# Print the updated DataFrame
# print("\nUsers DataFrame with 'name' column at the first position:")
# print(users)
merged_pivot = merged_data.pivot_table(
    index="name",
    columns=["closest_city", "age", "age_minimum", "artist"],
    values="merged_id",
    aggfunc="count",
    fill_value=0,
)
# print(merged_pivot)

# sparse matrix
merged_pivot_matrix = csr_matrix(merged_pivot.values)

# Display matrix
print("\nSparse Matrix:")
print(merged_pivot_matrix)

# KNN model with cosine similarity metric
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")

# Fit model using pivot table matrix
model_knn.fit(merged_pivot_matrix)
query_index = np.random.choice(merged_pivot.shape[0])

print("\nQuery Index:")
print(query_index)
print(f"Querying for: {merged_pivot.index[query_index]}")


# Get nearest neighbors (using the pivot table)
distances, indices = model_knn.kneighbors(
    merged_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6
)


# Print recommendations
print("\nRecommendations:")
for i in range(0, len(distances.flatten())):
    if i == 0:
        print(f"Recommendations for {merged_pivot.index[query_index]}:\n")
    else:
        print(
            f"{i}: {merged_pivot.index[indices.flatten()[i]]}, with distance of {distances.flatten()[i]:.4f}"
        )
