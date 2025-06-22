from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv("Mall_Customers.csv")  

df.drop('CustomerID', axis=1, inplace=True)

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_



plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=df['Cluster'], cmap='rainbow', s=50)
plt.title("Customer Segments (K = 6)")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.grid(True)
plt.show()

