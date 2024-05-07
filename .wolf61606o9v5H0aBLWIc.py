import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# Convert Excel sheets to CSV files
excel_file_path = "sample-data.xlsx"

excel_data = pd.read_excel(excel_file_path, sheet_name=None, engine="openpyxl")

for sheet_name, df in excel_data.items():
    csv_file_path = f"{sheet_name}.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"Sheet '{sheet_name}' has been converted to CSV file '{csv_file_path}'.")

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
    usecols=["date", "time", "location", "artist", "age_minimum"],
    dtype={
        "date": "str",
        "time": "str",
        "location": "str",
        "artist": "str",
        "age_minimum": "str",
    },
)

# Parse 'dob' column to datetime format
users["dob"] = pd.to_datetime(users["dob"])
performances["date"] = pd.to_datetime(performances["date"])
performances["time"] = pd.to
print(performances["date"])


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

# Print the updated DataFrame
print("\nUsers DataFrame with 'name' column at the first position:")
print(users)
