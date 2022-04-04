# SVM Classifier , acuratetea obtinuta pe Kaggle :0.63386
# mai jos import pachetele de care am nevoie de-a lungul implementarii
import sklearn
from sklearn import preprocessing
from sklearn import svm, metrics
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.image as image
import csv

#functie pentru normalizarea datelor dupa valoarea primita in parametru in type
def normalize(train_data, test_data, type = None):
    norm = None
    if type == 'standard':
        norm = preprocessing.StandardScaler()
    elif type == 'l1':
        norm = preprocessing.Normalizer('l1')   # Norma L1
    elif type == 'l2':
        norm = preprocessing.Normalizer('l2') # Norma L2
    if norm is not None:
        norm.fit(train_data)
        train_data = norm.transform(train_data)
        test_data = norm.transform(test_data)
    return train_data, test_data

# ---------------- train images
train_name_images = np.loadtxt('train.txt', 'str', delimiter=',', usecols=0) # citirea numelor imaginilor care se afla pe prima coloana din csv
train_classes = np.loadtxt('train.txt', 'int', delimiter=',', usecols=1) #citirea etichetelor care se afla pe a doua coloana din csv

train_images = [] #vor fi stocate imaginile sub forma de pixeli
for name in train_name_images: #parcurgem lista cu numele imaginilor, pentru a citi din fisierul ”train” fiecare imagine
    img = image.imread('train/' + name)
    arr = np.asarray(img)  # trasformam in array; imaginile au dimensiune 32x32
    arr = np.reshape(arr, (1, 1024)) #modificam dimensiunea array-ului din 32x32 in 1x1024 pentru a fi mai usor la procesare
    train_images.append(arr[0])

train_images = np.array(train_images) # trasformam lista in array

# Datele de validare si de testare sunt citite dupa modelul celor de antrenare
# ------------------ test images
test_name_images = np.loadtxt('test.txt', 'str', )

test_images = []
for name in test_name_images:
    img = image.imread('test/' + name)
    arr = np.asarray(img)
    arr = np.reshape(arr, (1, 1024))  # size ul unei imagini este 32x32
    test_images.append(arr[0])

test_images = np.array(test_images)

#------------------ validation images
validation_name_images = np.loadtxt('validation.txt', 'str', delimiter=',', usecols=0)
validation_classes = np.loadtxt('validation.txt', 'int', delimiter=',', usecols=1)

validation_images = []
for name in validation_name_images:
    img = image.imread('validation/' + name)
    arr = np.asarray(img)
    arr = np.reshape(arr, (1, 1024)) # size ul unei imagini este 32x32
    validation_images.append(arr[0])

validation_images = np.array(validation_images)


#normalizam datele de antrenare, de validare si de testare
norm_train, norm_test = normalize(train_images, test_images, "standard")
norm_train, norm_valid = normalize(train_images, validation_images, "standard")

#ne cream modelul
#de-a lungul implementarii am testat mai multe modele, dar pentru cel care nu se afla in comentarii am obtinut acuratetea cea mai mare
svm_model = svm.SVC(kernel = 'linear', C= 0.01)  #acuratetea 0.624
#svm_model = svm.SVC(kernel='rbf',gamma=0.5, C = 0.01) #Accuracy: 0.1108 peste 20 de min
#svm_model= svm.SVC(kernel = 'linear', C = 0.01) 0.4204, standardizare l1

svm_model.fit(norm_train, train_classes) # antrenam modelul

predictions_validation = svm_model.predict(norm_valid) #obtinem predictiile pentru datele de antrenare , cu ajutorul carora vom calcula acuratetea si matricea de confuzie
accuracy= accuracy_score(validation_classes, predictions_validation)
print("Accuracy score: " + str(accuracy))
print("Confusion matrix:")
matrix = confusion_matrix(validation_classes, predictions_validation)
print(matrix)

predictions = svm_model.predict(norm_test) # calculam predictiiile pentru datele de test
#vom scrie rezultatele intr-un fisier csv
# with open('sample_submission2.txt', mode='w', newline='') as file:
#     file_wr = csv.writer(file, delimiter=',')
#     file_wr.writerow(['id', 'label'])
#     for i in range(len(test_images)):
#        file_wr.writerow([test_name_images[i], predictions[i]])