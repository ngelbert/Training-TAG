import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, utils
from PIL import Image, ImageOps

# (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
pixelarea = 300*168
ndataset = 246
train_set = np.zeros((ndataset,pixelarea))
y_set = np.zeros(ndataset)
for i in range(ndataset):

    # creating an og_image object
    og_image = Image.open("nn_dataset/image" + str(i) + ".jpg")
    # og_image.show()
    #resize image to width of 300 with proper ratio 300 x 168
    basewidth = 300
    wpercent = (basewidth / float(og_image.size[0]))
    hsize = int((float(og_image.size[1]) * float(wpercent)))
    og_img = og_image.resize((basewidth, hsize), Image.ANTIALIAS)
    # img.save('nn_dataset/newimage0.jpg')
    # print(og_img.size)

    # applying grayscale method
    gray_image = ImageOps.grayscale(og_img)
    # gray_image.show()
    # print(gray_image.size)
    width, height = gray_image.size
    pixel_values = np.array(list(gray_image.getdata())) # convert mxn pixels (rn 1920 x 1080) to a 1d array of grayscale values size mn.
    # print(pixel_values)
    train_set[i] = pixel_values

with open('nn_dataset/y_file.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    j = 0
    for line in filecontents:
        for i in line.split():
            # print(int(i))
            y_set[j] = int(i)
        j = j+1
    filehandle.close()
# print(y_set)

X_train, X_test, y_train, y_test = train_test_split(train_set, y_set, test_size=0.2, random_state=1)
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

print("train x:", X_train)
print("train y:", y_train)

n_classes = 2
print("Shape before one-hot encoding: ", y_train.shape)
print("first 5 train y before one-hot")
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train.shape)
print("first 5 train y after one-hot:", y_train[:5])

# keras.backend.clear_session()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(pixelarea,)),
    keras.layers.Dense(units=1700, activation=tf.nn.relu, name="firstlayer", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1700, activation=tf.nn.relu, name="secondlayer", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    # keras.layers.Dense(units=1700, activation=tf.nn.relu, name="thirdlayer", kernel_regularizer=keras.regularizers.l2(0.0005)),
    # keras.layers.Dropout(0.3),
    # keras.layers.Dense(units=512, activation=tf.nn.relu, name="fourthlayer", kernel_regularizer=keras.regularizers.l2(0.0005)),
    # keras.layers.Dropout(0.4),
#     keras.layers.Dense(100, activation = tf.nn.relu, name="thirdlayer"),
    keras.layers.Dense(units=2, activation=tf.nn.softmax, name="outputlayer")
])

model.summary()

# sgd = keras.optimizers.SGD(lr=0.0001)
adam = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=2, validation_split=0.1)
# print(model.shape)
# print("first layer weights:", model.layers[1].get_weights()[0])
# print("first layer weights shape:", model.layers[1].get_weights()[0].shape)
# print("first layer biases:", model.layers[1].get_weights()[1])
# print("second layer weights:", model.layers[2].get_weights()[0])
# print("second layer biases:", model.layers[2].get_weights()[1])
# print("third layer weights:", model.layers[3].get_weights()[0])
# print("third layer biases:", model.layers[3].get_weights()[1])



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

y_pred = model.predict(X_test)

print("Y_test expected: ", y_test)
print("Y predicted values: ", y_pred)

countCorrect = 0.0
countTotal = float(len(y_pred))
for i in range(len(y_pred)):
    if (abs(y_test[i][0] - y_pred[i][0]) <= 0.5):
        countCorrect = countCorrect + 1.0
print("Test Acc: ", countCorrect / countTotal)

# sns.scatterplot(x=y_test, y=y_pred)
# plt.savefig('mse_nn.png')