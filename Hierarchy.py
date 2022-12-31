import numpy as np
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering

X=np.array([[5,3],
	[10,15],
	[15,12],
	[24,10],
	[30,30],
	[85,70],
	[71,80],
	[60,78],
	[70,55],
	[80,91],])

labels=range(1,11)
plt.scatter(X[:,0],X[:,1],label="True Postion")

for label,x,y in zip(labels, X[:,0],X[:,1]):
	plt.annotate(label,xy=(x,y),xytext=(-3,3),textcoords='offset points',ha='right',va='bottom')

plt.show()

linked=linkage(X,'single')
labelList=range(1,11)
dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=True)

plt.show()

cluster=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
cluster.fit_predict(X)
#plt.scatter(X[:,0],X[:,1],c=cluster.labels,cmap='rainbow')
plt.scatter(X[:,0],X[:,1],c=labels,cmap='rainbow')
plt.show()
