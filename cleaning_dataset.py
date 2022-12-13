import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
#from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function


#Task 1 Step 1
csv = pd.read_csv('loan_data.csv')
csv.drop_duplicates(subset ="id",keep = False, inplace = True)
csv = csv.dropna()



#Task 1 Step 2
columns_to_dummy = ['race', 'gender', 'type', 'approved'] #should we include date?
csv = csv.drop("id", axis=1)
csv = csv.drop("date", axis=1)


csv_dummy = pd.get_dummies(csv, columns=columns_to_dummy)
#print(csv_dummy)

label = csv['approved'].tolist()
#print(label)

csv_dummy.to_csv('clean_loan_data.csv')
#print(csv_dummy)



#Task 1 Step 3
#Splitting Training and Testing Data
train, test = train_test_split(csv_dummy, test_size=0.3)

train.to_csv("training.csv")
label_train = train['approved_True'].tolist()


test.to_csv("testing.csv")
label_test = test['approved_True'].tolist()




#Creating a Random Forrest Classifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(train, label_train)
rfc_pred=rfc.predict(test)

print("Random Forest")
#Accuracy
print("Accuracy of Random Forest:",
    metrics.accuracy_score(label_test, rfc_pred))
#Precision Score
print("Precision Score of Random Forest:",
    metrics.precision_score(label_test, rfc_pred, average='weighted'))
#Recall Score 
print("Recall Score of Random Forest:",
    metrics.recall_score(label_test, rfc_pred, average='weighted'))
#F1 Score
print("F1 Score of Random Forest:",
    metrics.f1_score(label_test, rfc_pred, average='weighted'))

#Confusion Matrix Random Forest
# Predict the test set
predictions = rfc.predict(test)

# Generate confusion matrix
#matrix = plot_confusion_matrix(rfc, test, label_test,cmap=plt.cm.Blues,normalize='true')
matrix = ConfusionMatrixDisplay.from_estimator(rfc, test, label_test)
plt.title('Confusion matrix for Random Forest Classifier')
#plt.show(matrix)
plt.show()




#Creating K Nearest Neighbors Classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train, label_train)
neigh_predict=neigh.predict(test)
#Confusion Matrix K Nearest Neighbors
# Predict the test set
neigh_pred = neigh.predict(test)


print()
print("K Nearest Neighbors")
#Accuracy
print("Accuracy of K Nearest Neighbors:",
    metrics.accuracy_score(label_test, neigh_pred))
#Precision Score
print("Precision Score of K Nearest Neighbors:",
    metrics.precision_score(label_test, neigh_pred, average='weighted'))
#Recall Score 
print("Recall Score of K Nearest Neighbors:",
    metrics.recall_score(label_test, neigh_pred, average='weighted'))
#F1 Score
print("F1 Score of K Nearest Neighbors:",
    metrics.f1_score(label_test, neigh_pred, average='weighted'))


# Generate confusion matrix
#matrix = plot_confusion_matrix(neigh, test, label_test,cmap=plt.cm.Blues,normalize='true')
matrix = ConfusionMatrixDisplay.from_estimator(neigh, test, label_test)
plt.title('Confusion matrix for K Nearest Neighbors Classifier')
#plt.show(matrix)
plt.show()


#Creating DecisionTree
# Create Decision Tree classifer object
dtc = DecisionTreeClassifier()
# Train Decision Tree Classifer
dtc = dtc.fit(train,label_train)
#Predict the response for test dataset
dtc_pred = dtc.predict(test)

#Confusion Matrix DecisionTree Classifier
# Predict the test set
predictions = dtc.predict(test)

print()
print("Decision Tree")
#Accuracy
print("Accuracy of Decision Tree:",
    metrics.accuracy_score(label_test, dtc_pred))
#Precision Score
print("Precision Score of Decision Tree:",
    metrics.precision_score(label_test, dtc_pred, average='weighted'))
#Recall Score 
print("Recall Score of Decision Tree:",
    metrics.recall_score(label_test, dtc_pred, average='weighted'))
#F1 Score
print("F1 Score of Decision Tree:",
    metrics.f1_score(label_test, dtc_pred, average='weighted'))
#Top K Accuracy
print("Top K Accuracy Score of Decision Tree:",
    metrics.top_k_accuracy_score(label_test, dtc_pred, k=1))
#Jaccard Score
print("Jaccard Score of Decision Tree:",
    metrics.jaccard_score(label_test, dtc_pred))

# Generate confusion matrix
#matrix = plot_confusion_matrix(dtc, test, label_test,cmap=plt.cm.Blues,normalize='true')
matrix = ConfusionMatrixDisplay.from_estimator(dtc, test, label_test)
plt.title('Confusion matrix for Decision Tree Classifier')
#plt.show(matrix)
plt.show()

#Task 1 Step 4
#Decision Tree Classifier is my chosen final model.