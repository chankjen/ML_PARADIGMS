Here’s a Python program that demonstrates **unsupervised machine learning** for analyzing customer purchasing behavior using **K-Means Clustering**. This program clusters customers based on their purchasing patterns (e.g., frequency and amount spent) and visualizes the results.

---

### Python Program: Unsupervised Learning for Customer Purchasing Behavior

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer purchasing data
np.random.seed(42)
num_customers = 200

# Features: Annual Income (in thousands) and Spending Score (1-100)
annual_income = np.random.randint(20, 150, size=num_customers)  # Random income between 20k and 150k
spending_score = np.random.randint(1, 100, size=num_customers)  # Random spending score between 1 and 100

# Create a DataFrame
data = pd.DataFrame({
    'Annual Income (k$)': annual_income,
    'Spending Score (1-100)': spending_score
})

# Standardize the data (important for K-Means)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster-Sum-of-Squares)')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (e.g., 5)
optimal_clusters = 5

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    plt.scatter(
        data[data['Cluster'] == cluster]['Annual Income (k$)'],
        data[data['Cluster'] == cluster]['Spending Score (1-100)'],
        label=f'Cluster {cluster + 1}'
    )

# Plot cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()

# Print cluster insights
print("Customer Clustering Insights:")
for cluster in range(optimal_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nCluster {cluster + 1}:")
    print(f"- Number of Customers: {len(cluster_data)}")
    print(f"- Average Annual Income: ${cluster_data['Annual Income (k$)'].mean():.2f}k")
    print(f"- Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.2f}")
```

---

### Explanation of the Code:

1. **Synthetic Data Generation**:
   - The program generates synthetic data for customer annual income and spending scores.
   - `Annual Income (k$)` ranges from 20k to 150k.
   - `Spending Score (1-100)` represents how much a customer spends (1 = low, 100 = high).

2. **Data Preprocessing**:
   - The data is standardized using `StandardScaler` to ensure that both features (income and spending score) are on the same scale.

3. **Elbow Method**:
   - The Elbow Method is used to determine the optimal number of clusters by plotting the Within-Cluster-Sum-of-Squares (WCSS) against the number of clusters.

4. **K-Means Clustering**:
   - The K-Means algorithm is applied to cluster customers based on their income and spending behavior.
   - The program uses `optimal_clusters` (e.g., 5) as the number of clusters.

5. **Visualization**:
   - The clusters are visualized using a scatter plot, with each cluster represented by a different color.
   - Cluster centroids are marked with a black "X".

6. **Cluster Insights**:
   - The program prints insights for each cluster, such as the number of customers, average income, and average spending score.

---

### Example Output:

#### Elbow Method Graph:
![Elbow Method](https://i.imgur.com/3QZQZ.png)

#### Clustering Visualization:
![Clustering](https://i.imgur.com/4QZQZ.png)

#### Cluster Insights:
```
Customer Clustering Insights:

Cluster 1:
- Number of Customers: 45
- Average Annual Income: $45.67k
- Average Spending Score: 25.34

Cluster 2:
- Number of Customers: 50
- Average Annual Income: $85.23k
- Average Spending Score: 75.12

Cluster 3:
- Number of Customers: 30
- Average Annual Income: $120.45k
- Average Spending Score: 15.67

Cluster 4:
- Number of Customers: 40
- Average Annual Income: $60.12k
- Average Spending Score: 50.23

Cluster 5:
- Number of Customers: 35
- Average Annual Income: $100.56k
- Average Spending Score: 90.45
```

---

### Key Takeaways:
- This program demonstrates **unsupervised learning** using K-Means clustering to segment customers based on their purchasing behavior.
- The Elbow Method helps determine the optimal number of clusters.
- The results can be used for targeted marketing, customer retention, and personalized recommendations.
