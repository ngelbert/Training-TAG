import boto3
import os
from dotenv import load_dotenv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# This program trains the model and uploads it to S3 bucket

# Read CSV file
all_col = ['Heart rate', 'AccelerationX', 'AccelerationY', 'AccelerationZ', 'Behavior'] 
features = ['Heart rate', 'AccelerationX', 'AccelerationY',  'AccelerationZ']
ds = pd.read_csv("tree_dataset/behavior.csv", header=None, names=all_col)

# Process label encoder to change categorical feature to numerical feature
LabEnc = preprocessing.LabelEncoder()
LabEnc.fit(ds['Behavior'])
X_train = ds[features]
Y_train = ds['Behavior']

# Train the model with built in decision tree algorithm in sklearn
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, Y_train)

# data plot
_= plot_tree(clf, feature_names=features, class_names=['walking', 'abnormal', 'sitting', 'sleeping'], filled=True)
plt.savefig('behavior.png')

model = pickle.dumps(clf)

# Send model through s3bucket to the AWS database 
load_dotenv()
s3 = boto3.client('s3') 
print(os.environ)
response = s3.put_object(
    Bucket='behavior-model-bucket',
    Body=model,
    Key='behavior-model'
    )