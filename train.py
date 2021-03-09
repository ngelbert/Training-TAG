from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
all_col = ['Heart rate', 'Acceleration', 'Temperature', 'Behavior'] 
features = ['Heart rate', 'Acceleration', 'Temperature']
ds = pd.read_csv("behavior.csv", header=None, names=all_col)

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
# print("Mean Squared Error (MSE): %.2f" % mean_squared_error(Y_test, Y_pred))
# print("Coefficient of Determination (R^2): %.2f" % r2_score(Y_test, Y_pred))

print("Y_test expected: ", Y_test)
print("Y predicted values: ", Y_pred)

# sns.scatterplot(x=Y_test, y=Y_pred)
# plt.show()

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = features,class_names=['walking', 'running', 'sitting', 'sleeping'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('behavior.png')
# Image(graph.create_png())
# fig = plt.figure(figsize(25,20))
_= plot_tree(clf, feature_names=features, class_names=['walking', 'running', 'sitting', 'sleeping'], filled=True)
plt.savefig('behavior.png')

# decision tree algoo/ random tree