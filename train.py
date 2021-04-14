from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# File to test model through train and test split (Predict test data)

# Read CSV file
all_col = ['Heart rate', 'AccelerationX', 'AccelerationY', 'AccelerationZ', 'Behavior'] 
features = ['Heart rate', 'AccelerationX', 'AccelerationY',  'AccelerationZ']
ds = pd.read_csv("tree_dataset/behavior.csv", header=None, names=all_col)

#Process label encoder to change categorical feature to numerical feature
LabEnc = preprocessing.LabelEncoder()
LabEnc.fit(ds['Behavior'])
X= ds[features]
Y= ds['Behavior']

#80/20 train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Train the model with built in decision tree algorithm in sklearn
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, Y_train)

#predict model
Y_pred = clf.predict(X_test)

#show results
print("Accuracy:", accuracy_score(Y_test, Y_pred))

print("Y_test expected: ", Y_test)
print("Y predicted values: ", Y_pred)

_= plot_tree(clf, feature_names=features, class_names=['walking', 'abnormal', 'sitting', 'sleeping'], filled=True)
plt.savefig('behavior.png')

# decision tree algoo/ random tree