# CNN Classifier , acuratetea obtinuta pe Kaggle :0.86960
#mai jos import pachetele de care am nevoie de-a lungul implementarii
import tensorflow as tf
from keras.layers import BatchNormalization
from tensorflow import keras
import sklearn
from sklearn import preprocessing
from sklearn import svm, metrics
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.image as image
import csv

#---------------- train images
train_name_images = np.loadtxt('train.txt', 'str', delimiter=',', usecols=0)# citirea numelor imaginilor care se afla pe prima coloana din csv
train_classes = np.loadtxt('train.txt', 'int', delimiter=',', usecols=1)  #citirea etichetelor care se afla pe a doua coloana din csv
nr_classes = len(set(train_classes)) # 9 clase

train_images = [] #vor fi stocate imaginile sub forma de pixeli
for name in train_name_images: #parcurgem lista cu numele imaginilor, pentru a citi din fisierul ”train” fiecare imagine
    img = image.imread('train/' + name)
    arr = np.asarray(img) # trasformam in array; imaginile au dimensiune 32x32
    train_images.append(arr)


train_images = np.array(train_images)

# Datele de validare si de testare sunt citite dupa modelul celor de antrenare
# ------------------ test images
test_name_images = np.loadtxt('test.txt', 'str', )

test_images = []
for name in test_name_images:
    img = image.imread('test/' + name)
    arr = np.asarray(img)
    test_images.append(arr)

test_images = np.array(test_images)

#------------------ validation images
validation_name_images = np.loadtxt('validation.txt', 'str', delimiter=',', usecols=0)
validation_classes = np.loadtxt('validation.txt', 'int', delimiter=',', usecols=1)

validation_images = []
for name in validation_name_images:
    img = image.imread('validation/' + name)
    arr = np.asarray(img)
    validation_images.append(arr)

validation_images = np.array(validation_images)

#normalizam datele
train_mean = train_images.mean(axis=0, keepdims=True)  #calculam media
train_std = train_images.std(axis=0, keepdims=True)  #calculam deviatia standard
norm_train = (train_images - train_mean) / train_std #aplicam formula pentru normalizare
norm_valid = (validation_images - train_mean) / train_std #normalizam datele de validare
norm_test = (test_images - train_mean) / train_std #normalizam datele de testare


norm_train = norm_train[..., np.newaxis] #adaugam o noua dimensiune de lungime 1 pt pozitii
norm_valid = norm_valid[..., np.newaxis]
norm_test = norm_test[..., np.newaxis]

#crearea modelului cu layerele specifice
model = keras.models.Sequential([
                                 keras.layers.Conv2D(32, 5, activation="relu", padding= "same", input_shape=[32, 32, 1]), # input shape ul asa pentru ca avem
                                 keras.layers.MaxPool2D(2),                                                             # imagini de dimensiuni 32x32 si alb-negru(1)

                                 keras.layers.Conv2D(32, 5, activation="relu", padding="same"),
                                 keras.layers.MaxPool2D(2),
                                 BatchNormalization(),

                                 keras.layers.Conv2D(128, 5, activation="relu", padding="same"),
                                 keras.layers.MaxPool2D(2),
                                 BatchNormalization(),

                                 keras.layers.Flatten(),

                                 keras.layers.Dense(100, activation="relu"),
                                 keras.layers.Dropout(0.25),

                                 keras.layers.Dense(10, activation="softmax")
                                 ])

# model.compile(loss = "sparse_categorical_crossentropy",   #am obtinut o acuratete mai buna cu a doua optimizare
# optimizer = "adam",
# metrics = ["accuracy"])
model.compile(  #compilam modelul
  optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
  loss= "sparse_categorical_crossentropy",
  metrics=['accuracy'])

#antrenam modelul, ii dam datele de antrenare si numarul de epoci
history = model.fit(norm_train,
                    train_classes,
                    epochs= 2)
#evaluam rezultatul obtinut pe datele de validare
test_loss, test_acc = model.evaluate(norm_valid, validation_classes, verbose=2)

print("Test Loss:" + str(test_loss))
print("Test Accuracy: "+str(test_acc))

#calculam matricea de confuzie
predictions_validation = model.predict(norm_valid)
validations = np.zeros(len(predictions_validation))
for i in range(len(predictions_validation)):
    validations[i] = np.argmax(predictions_validation[i])
matrix = confusion_matrix(validation_classes, validations)
print("Confusion matrix:")
print(matrix)

#scriem in fisierul csv predictiile obtinute
# predictions = model.predict(norm_test)
# with open('sample_submission5.txt', mode='w', newline='') as file:
#     file_wr = csv.writer(file, delimiter=',')
#     file_wr.writerow(['id', 'label'])
#     for i in range(len(test_images)):
#        file_wr.writerow([test_name_images[i], np.argmax(predictions[i])])
#

