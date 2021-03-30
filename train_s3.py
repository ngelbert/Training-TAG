import boto3
import os
from dotenv import load_dotenv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

# This program trains the model and uploads it to S3 bucket

# Read CSV file
all_col = ['Heart rate', 'Acceleration', 'Temperature', 'Behavior'] 
features = ['Heart rate', 'Acceleration', 'Temperature']
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
model = pickle.dumps(clf)

# Remember to store AWS keys on env, don't push .env to remote
load_dotenv()
s3 = boto3.client('s3') 
print(os.environ)
response = s3.put_object(
    Bucket='behavior-model-bucket',
    Body=model,
    Key='behavior-model'
    )