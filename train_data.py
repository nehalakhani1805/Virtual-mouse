import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
# from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import tensorflow as tf
import keras.backend as K
# df=pd.read_csv('train_foo.csv')
# df=df.dropna()
# df.to_csv('train_foo2.csv')
# print(df.isna().sum())
# print("Shape")
# print(df.shape)
def keras_model(image_x, image_y):
    num_of_classes = 6
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "D:/virtual_mouse/Virtual-Mouse/trained_model.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def main():
    data = pd.read_csv("D:/virtual_mouse/Virtual-Mouse/train_foo2.csv")
    dataset = np.array(data)
    np.random.shuffle(dataset)
    X = dataset
    Y = dataset
    X = X[:, 1:2501]
    Y = Y[:, 0]

    X_train = X[0:6000, :]
    X_train = X_train / 255.
    X_test = X[6000:7201, :]
    X_test = X_test / 255.

    # Reshape
    Y = Y.reshape(Y.shape[0], 1)
    Y_train = Y[0:6000, :]
    Y_train = Y_train.T
    Y_test = Y[6000:7201, :]
    Y_test = Y_test.T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    # Y_train=Y_train-1
    print(Y_train[0])
    image_x = 50
    image_y = 50
   
    # Y_train_a=tf.keras.backend.get_value(Y_train)
    # train_y = np_utils.to_categorical(Y_train)
    # Y_test_a=tf.keras.backend.get_value(Y_test)
    # test_y = np_utils.to_categorical(Y_test)
    
    train_y = to_categorical(Y_train)
    test_y = to_categorical(Y_test)
    train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
    test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
    print("X_train shape: " + str(X_train.shape))
    print("X_test shape: " + str(X_test.shape))

    model, callbacks_list = keras_model(image_x, image_y)
    model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=10, batch_size=64,callbacks=callbacks_list)
    scores = model.evaluate(X_test, test_y, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('D:/virtual_mouse/Virtual-Mouse/trained_model.h5')


main()