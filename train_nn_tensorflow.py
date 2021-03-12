import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = X_train / 255.0
# X_train_sub, temp1, y_train_sub, temp2 = train_test_split(X_train, y_train, test_size=0.9)#split to only 6000  
print("train x:", X_train)
print("train y:", y_train)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

keras.backend.clear_session()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(1000, activation=tf.nn.relu, name="firstlayer"),
    keras.layers.Dense(1000, activation=tf.nn.relu, name="secondlayer"),
#     keras.layers.Dense(100, activation = tf.nn.relu, name="thirdlayer"),
    keras.layers.Dense(10, activation=tf.nn.relu, name="outputlayer")
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, verbose=2)
print("first layer weights:", model.weights[0])
# print("first layer bias:", model.bias[0])

print(history.history)
loss_train = history.history['loss']
# loss_val = trained_model.history['val_loss']
epochs = range(1,6)
plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()