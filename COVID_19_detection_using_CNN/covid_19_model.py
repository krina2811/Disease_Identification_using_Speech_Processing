import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def load_dataset(dataset_path):

    with open(dataset_path, 'r') as fr:
        data = json.load(fr)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def prepare_dataset(test_size, val_size):

    #load dataset
    X, y = load_dataset('covid_19_data1.json')

    #split into train_test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)

    #split into train and validation set
    X_train, X_validation, y_train, y_validation  = train_test_split(X_train, y_train, test_size= val_size)

    #CNN take 4d array convert it into 4d array -> (number_Samples, 100, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding= "same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding= "same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128, (2,2), activation="relu"))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding= "same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(7, activation="softmax"))
    return model

def predict(model, X, y):

    
    # X is 3 dimentional array convert it into 4 dimention array
    X = X[np.newaxis, ...]

    # predict return array with probability
    prediction = model.predict(X)   

    # extract the index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("The Expected index : {}, Predicted index : {}".format(y, predicted_index))


def plot_graph(epochs, history):
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    
    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
    
    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
    
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("E:\\audio_classification\\res_cnn2\\covid_plot.png")


if __name__ == "__main__":

    # create train, validation and test set
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)
    # print(len(X_train))  # 10334
    # print(len(X_test))   # 4304
    # print(len(X_validation)) #2584

    # print(X_test.shape)
    # exit()
    #build model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)
    # model.summary()

    # #compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer= optimizer, loss = "sparse_categorical_crossentropy",
                metrics = ['accuracy'])

    # #train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=64, epochs=75)


    ## save model
    model_path = 'covid1.h5'
    # model.save(model_path)

    # evaluate CNN on test set
    # test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    # print("Accuracy on test set is: {}".format(test_accuracy))

    # # plot accuracy graph
    plot_graph(75, history)

    ##load model
    # model = load_model(model_path)
    yhat_classes = model.predict_classes(X_test, verbose=0)
    print(yhat_classes)
    print(y_test)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes, average='macro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes, average='macro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes, average='macro')
    print('F1 score: %f' % f1)
    
    
    # test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    # print("Accuracy on test set is: {}".format(test_accuracy))

    #make prediction on single sample
    X = X_test[457]
    y = y_test[457]

    predict(model, X, y)