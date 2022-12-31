from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt 

iris=load_iris()
x=iris.data
y=iris.target

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=24)
scores={}
scores_list=[]
krange=range(1,26)
for k in krange:
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(xtrain,ytrain)
	ypred=knn.predict(xtest)
	scores[k]=metrics.accuracy_score(ytest,ypred)
	scores_list.append(metrics.accuracy_score(ytest,ypred))

plt.plot(krange,scores_list)
plt.xlabel('Values of K for the KNN')
plt.ylabel('Testing accuracy')
plt.show()