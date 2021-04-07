import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, utils
from PIL import Image, ImageOps
import pickle

pixelarea = 300*168
ndataset = 246
train_set = np.zeros((ndataset,pixelarea))
y_set = np.zeros(ndataset)
for i in range(ndataset):

    # creating an og_image object
    og_image = Image.open("nn_dataset/image" + str(i) + ".jpg")
    
    # resize image to width of 300 with proper ratio 300 x 168
    basewidth = 300
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

# make x_train range from 0 to 1 for proportionality
X_train = train_set
y_train = y_set
X_train = X_train.astype('float32')
X_train = X_train / 255.0

#convert y to one hot for multi labeled classifier
n_classes = 2
y_train = utils.to_categorical(y_train, n_classes)

# CNN model (Convoluted Neural Network) using 2 hidden layers each having 1700 nodes. 
# Model uses ReLu activation function, and the softmax activation function at the output layer. 
# Dropout of 0.5 and l2 regularizer of coefficient 0.001 is used to prevent overfitting
# Model is at LR of 0.0005 with the adam optimizer (LR decay) 
# Model uses cross entropy loss to calculate for the loss in training and validation
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(pixelarea,)),
    keras.layers.Dense(units=1700, activation=tf.nn.relu, name="firstlayer", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1700, activation=tf.nn.relu, name="secondlayer", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=2, activation=tf.nn.softmax, name="outputlayer")
])

model.summary()

adam = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model is trained with batch_size of 128 and through a total of 40 epochs, with 10% of the training data used as validation set.

history = model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=2, validation_split=0.1)

# plotting graph of training and validation loss over the number of epochs to see the model

print(history.history)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,31)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_nn_train.png')

pickled_model = pickle.dumps(clf)

# AWS associated code to store model in the back-end