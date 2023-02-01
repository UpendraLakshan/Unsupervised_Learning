from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
import pandas as pd

X,y=make_blobs(n_samples=300,cluster_std=1.00,random_state=12)

plt.scatter(X[:,0],X[:,1])
plt.show()

wcss=[]
for i in range(1,11):
	kmeans=KMeans(n_clusters=i)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=3)
pred_y=kmeans.fit_predict(X)
plt.scatter(X[:,0],X[:,1],cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red')
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_.astype(float))

print('predY',pred_y)

plt.show()
