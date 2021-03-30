import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, utils
from PIL import Image, ImageOps

# (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
pixelarea = 1280*720
ndataset = 1
train_set = np.zeros((ndataset,pixelarea))
for i in range(ndataset):

    # creating an og_image object
    og_image = Image.open("nn_dataset/trainimg" + str(i) + ".jpg")
    # og_image.show()

    # applying grayscale method
    gray_image = ImageOps.grayscale(og_image)
    # gray_image.show()
    print(gray_image.size)
    width, height = gray_image.size
    pixel_values = np.array(list(gray_image.getdata())) # convert mxn pixels (rn 1280 x 720) to a 1d array of grayscale values size mn.
    print(pixel_values)
    train_set[i] = pixel_values
print(train_set)



# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# print("train x:", X_train.shape)
# print("train y:", y_train)

# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# print("first 5 train y before one-hot")
# y_train = utils.to_categorical(y_train, n_classes)
# y_test = utils.to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", y_train.shape)
# print("first 5 train y after one-hot:", y_train[:5])
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # keras.backend.clear_session()
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(784,)),
#     keras.layers.Dense(units=512, activation=tf.nn.relu, name="firstlayer"),
#     keras.layers.Dense(units=512, activation=tf.nn.relu, name="secondlayer"),
#     keras.layers.Dropout(0.2),
# #     keras.layers.Dense(100, activation = tf.nn.relu, name="thirdlayer"),
#     keras.layers.Dense(units=10, activation=tf.nn.softmax, name="outputlayer")
# ])

# model.summary()

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=2, validation_split=0.1)
# print("first layer weights:", model.weights[0])

# print(history.history)
# loss_train = history.history['loss']
# loss_val = history.history['val_loss']
# epochs = range(1,6)
# plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='validation loss')
# plt.title('Training loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('loss_nn_train.png')

# y_pred = model.predict(X_test)

# print("Y_test expected: ", y_test)
# print("Y predicted values: ", y_pred)

# sns.scatterplot(x=y_test, y=y_pred)
# plt.savefig('mse_nn.png')