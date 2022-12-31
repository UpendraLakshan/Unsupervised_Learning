import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_excel("Data.xlsx")
data.head()


cols = list(data.columns.values)
cols.pop(cols.index('A'))
data = data[cols+['A']]
cols = list(data.columns.values)
cols.pop(cols.index('B'))
data = data[cols+['B']]
cols = list(data.columns.values)
cols.pop(cols.index('C'))
data = data[cols+['C']]
data.head()

X = data.iloc[:,2:4].values

#elbow method
wcss = []
for i in range(1,6):
	print(i)
	k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
	k_means.fit(X)
	wcss.append(k_means.inertia_)
#plot elbow curve
plt.plot(np.arange(1,6),wcss)
plt.xlabel('Clusters')
plt.ylabel('SSE')
plt.show()


k_means_optimum = KMeans(n_clusters = 3, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(X)
print(y)

data['cluster'] = y  
# the above step adds extra column indicating the cluster number for each country

data1 = data[data.cluster==0]
data2 = data[data.cluster==1]
data3 = data[data.cluster==2]

kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)
kplot.plot3D(xline, yline, zline, 'black')
# Data for three-dimensional scattered points
kplot.scatter3D(data1.A, data1.B, data1.C, c='red', label = 'Cluster 1')
kplot.scatter3D(data2.A,data2.B,data2.C,c ='green', label = 'Cluster 2')
kplot.scatter3D(data3.A,data3.B,data3.C,c ='blue', label = 'Cluster 3')
#plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 200)
plt.legend()
plt.title("Kmeans")
plt.show()