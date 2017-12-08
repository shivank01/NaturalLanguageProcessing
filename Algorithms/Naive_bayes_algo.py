from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.metrics import confusion_matrix

import numpy as np

iris=datasets.load_iris()

gnb=GaussianNB()
mnb=MultinomialNB()

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

gnb.fit(X_train,y_train)

g_accuracy=gnb.score(X_test,y_test)

print("gaussian accuracy:",g_accuracy)

mnb.fit(X_train,y_train)

m_accuracy=mnb.score(X_test,y_test)

print("multinomial accuracy:",m_accuracy)
	
ar=[4.9,3.0,1.4,0.2]
p=np.array(ar).reshape(1,-1)

print("gaussian prediction:",gnb.predict(p))

print("multinomial prediction:",mnb.predict(p))


y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
cnf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)

print("gaussian confusion matrix:\n",cnf_matrix_gnb)

y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)
cnf_matrix_mnb = confusion_matrix(y_test, y_pred_mnb)
print("multinomial confusion matrix:\n",cnf_matrix_mnb)