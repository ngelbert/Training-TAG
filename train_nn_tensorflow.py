import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, utils
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import json

pixelarea = 50*28
ndataset = 246
train_set = np.zeros((ndataset,pixelarea))
y_set = np.zeros(ndataset)
for i in range(ndataset):

    # creating an og_image object
    og_image = Image.open("nn_dataset/image" + str(i) + ".jpg")
    
    # resize image to width of 50 with proper ratio 50 x 28
    basewidth = 50
    wpercent = (basewidth / float(og_image.size[0]))
    hsize = int((float(og_image.size[1]) * float(wpercent)))
    og_img = og_image.resize((basewidth, hsize), Image.ANTIALIAS)

    # applying grayscale method
    gray_image = ImageOps.grayscale(og_img)
    width, height = gray_image.size
    pixel_values = np.array(list(gray_image.getdata())) # convert mxn pixels (rn 1920 x 1080) to a 1d array of grayscale values size mn.
    train_set[i] = pixel_values

# open the associated y output file and write contents to an array
with open('nn_dataset/y_file.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    j = 0
    for line in filecontents:
        for i in line.split():
            y_set[j] = int(i)
        j = j + 1
    filehandle.close()

# make x_train, x_test range from 0 to 1 for proportionality
X_train, X_test, y_train, y_test = train_test_split(train_set, y_set, test_size=0.2, random_state=1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

#convert y to one hot for multi labeled classifier
n_classes = 2
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)

# CNN model (Convoluted Neural Network) using 3 hidden layers each having 900 nodes. 
# Model uses ReLu activation function, and the softmax activation function at the output layer. 
# Dropout of 0.5 and l2 regularizer of coefficient 0.01 is used to prevent overfitting
# Model is at LR of 0.004 with the adam optimizer (LR decay) 
# Model uses cross entropy loss to calculate for the loss in training and validation
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(pixelarea,)),
    keras.layers.Dense(units=900, activation=tf.nn.relu, name="firstlayer", kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=900, activation=tf.nn.relu, name="secondlayer", kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=900, activation=tf.nn.relu, name="thirdlayer", kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=2, activation=tf.nn.softmax, name="outputlayer")
])

model.summary()

adam = keras.optimizers.Adam(learning_rate=0.004)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model is trained with batch_size of 64 and through a total of 40 epochs, with 10% of the training data used as validation set.

history = model.fit(X_train, y_train, batch_size=64, epochs=40, verbose=2, validation_split=0.1)

# Weights are given by a (next layer node x previous layer node) size matrix.
# biases are given by a (Next layer node) size vector
print("first layer weights:", model.layers[1].get_weights()[0])
print("first layer biases:", model.layers[1].get_weights()[1])
print("second layer weights:", model.layers[3].get_weights()[0])
print("second layer biases:", model.layers[3].get_weights()[1])
print("third layer weights:", model.layers[5].get_weights()[0])
print("third layer biases:", model.layers[5].get_weights()[1])
print("output layer weights:", model.layers[7].get_weights()[0])
print("output layer biases:", model.layers[7].get_weights()[1])

dup = {}
dup['firstlayer'] = {
    'weights': model.layers[1].get_weights()[0].tolist(),
    'biases': model.layers[1].get_weights()[1].tolist(),
}
dup['secondlayer'] = {
    'weights': model.layers[3].get_weights()[0].tolist(),
    'biases': model.layers[3].get_weights()[1].tolist(),
}
dup['thirdlayer'] = {
    'weights': model.layers[5].get_weights()[0].tolist(),
    'biases': model.layers[5].get_weights()[1].tolist(),
}
dup['fourthlayer'] = {
    'weights': model.layers[7].get_weights()[0].tolist(),
    'biases': model.layers[7].get_weights()[1].tolist(),
}
with open('nn.json', 'w') as outfile:
    json.dump(dup, outfile)

# plotting graph of training and validation loss over the number of epochs to see the model

print(history.history)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,41)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_nn_train.png')

# Predict model
y_pred = model.predict(X_test)

print("Y_test expected: ", y_test)
print("Y predicted values: ", y_pred)

# Accuracy score
countCorrect = 0.0
countTotal = float(len(y_pred))
for i in range(len(y_pred)):
    if (abs(y_test[i][0] - y_pred[i][0]) <= 0.5):
        countCorrect = countCorrect + 1.0
print("Test Acc: ", countCorrect / countTotal)
