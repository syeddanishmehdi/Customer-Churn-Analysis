# Customer Churn Clustering Analysis

This project performs K-means clustering on customer data to segment users based on their usage and income characteristics. The analysis helps identify different customer profiles, which can be used for targeted marketing or retention strategies.

## Features

- Loads and preprocesses customer churn data
- Drops non-numeric (object) columns for clustering
- Sets customer ID as the DataFrame index
- Converts all features to `float64` for compatibility
- Applies K-means clustering to segment customers into 4 groups
- Outputs the size and centroid of each cluster for interpretation

## Prerequisites

- Python 3.7+
- pandas
- scikit-learn
- numpy

Install dependencies with:
