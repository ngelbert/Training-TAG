import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#data may be read from csv
X_train, Y_train, X_test, Y_test = train_test_split(x,y,test_size=0.2)

model = MLPRegressor(hidden_layer_sizes=(1000,5), activation="relu", solver='adam',
                     learning_rate_init=0.01, max_iter=1500, verbose=True) # set at 1000 hidden nodes for 5-layer DNN, ReLU activation function, adam gradient descent method, at 1500 epochs

model = model.fit(X_train,Y_train) #train the model with forwardfeed + backprop

param = model.get_params(deep=True)
print("Loss: ", model.loss_)

model.predict(X_test)

