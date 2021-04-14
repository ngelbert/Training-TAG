# Training-TAG
#
# This repository contains all the python files, dataset folders and output/loss graphs of the Machine Learning model
# There are two Machine Learning models implemented:

# 1. Machine Learning with Decision Trees
# This model is a decision tree that takes in accelerometer and heart rate data to predict the animal behaviour based on the data given
# There are 4 types of animal behavior: Sleeping, Sitting, Walking and abnormal
# The current model used and stored on the cloud can be seen in the file behavior.png
# The dataset used to train the model are 400 rows of data stored in a csv file (tree_dataset/behavior.csv)
# The testing file used to calibrate and optimize the code running the model is in train.py, and the actual file run to send the model to the AWS serverless back-end is train_s3.py