import pandas as pd                   # Import pandas for data manipulation
from sklearn.cluster import KMeans    # Import KMeans for clustering
import numpy as np                    # Import numpy (not used in this script, but often useful)

# Load the data from a CSV file named 'churn.txt'
df = pd.read_csv("churn.txt")

# 1. Print variable (column) names and their data types
print("Variable Names and Types:")
print(df.dtypes)

# 2. Identify columns with object (string) data types and drop them from the DataFrame
object_cols = df.select_dtypes(include=['object']).columns
df = df.drop(object_cols, axis=1)
print("\nDataFrame after dropping object columns:")
print(df.head())

# 3. Set the 'ID' column as the DataFrame index for easier data management
df = df.set_index('ID')
print("\nDataFrame with 'ID' as index:")
print(df.head())

# 4. Convert all columns to float64 data type for numerical processing
for col in df.columns:
    df[col] = df[col].astype('float64')
print("\nDataFrame with float64 data types:")
print(df.dtypes)

# 5. Apply K-means clustering with 4 clusters and assign cluster labels to a new column
kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')
df['Cluster'] = kmeans.fit_predict(df)

# 6. Print the number of members in each cluster
print("\nNumber of members in each cluster:")
print(df['Cluster'].value_counts())

# 7. Calculate and print the mean (centroid) of each cluster for interpretation
centroids = df.groupby('Cluster').mean()
print("\nCluster Centroids:")
print(centroids)

# Example cluster descriptions (adjust based on your data analysis)
# Cluster 0: Has generally lower usage and income
# Cluster 1: Has mid-range usage and income
# Cluster 2: Has high usage and income
# Cluster 3: Has very low usage and income, potentially new or inactive users
