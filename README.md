# Training-TAG
## This repository contains all the python files, dataset folders and output/loss graphs of the Machine Learning model
There are two **Machine Learning** models implemented:

#### 1. Machine Learning with Decision Trees

This model is a decision tree that takes in accelerometer and heart rate data to predict the animal behaviour based on the data given. There are 4 types of animal behavior: Sleeping, Sitting, Walking and abnormal. The current model used and stored on the cloud can be seen in the file *behavior.png*. The dataset used to train the model are 400 rows of data stored in a csv file *tree_dataset/behavior.csv*. The testing file used to calibrate and optimize the code running the model is in *train.py*, and the actual file run to send the model to the AWS serverless back-end is *train_s3.py*.

#### 2. Machine Learning with Neural Network

This model is a discrete neural network that takes in an image to predict whether there is a human or not. The model is mainly used to prevent poaching in wildlife. The model consists of 8 layers, where there is a Flatten input layer, 4 hidden layers with 900 nodes each, 3 dropout layers and 1 output layer with 2 nodes. The hidden layers have a regularizer of 0.01 each and together with the dropout layer, this prevents the model from overfitting. The output of the model is a one-hot code, where [1 0] means there are no human nearby, and [0 1] means otherwise. The model is trained using the 246 images in the folder nn_dataset where its corresponding expected outputs are in the file *nn_dataset/y_file.txt*, and the images are first compressed and grayscaled as shown in the folder *nn_dataset_comp_gray* before being fed into the model for training and testing. The testing file used to calibrate and optimize the code running the model is in *train_nn_tensorflow.py*, while the actual code used to send the weights and biases to the fpga is in *train_nn_fpga.py*. The file *train_nn_s3.py* is used as a backup that sends the model to the AWS backend in case the fpga does not work. The weights and biases of the current model is saved into the *nn.json* file, and the training and validation loss over the epochs graph is in *loss_nn_train.png*

**Note that the .env file is not pushed to the repo just for privacy reasons. This file stores the AWS private keys to send data over to the databse.**