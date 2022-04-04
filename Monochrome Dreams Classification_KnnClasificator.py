# Knn Classifier , acuratetea obtinuta pe Kaggle :0.50426
#mai jos import pachetele de care am nevoie de-a lungul implementarii
import numpy as np
import math
import matplotlib.image as image
import csv
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, confusion_matrix

#am creat clasa clasificatorului
class KnnClassifier:
    def __init__(self, train_images, train_classes):
        self.train_images = train_images
        self.train_classes = train_classes

    def classify(self, test_images, num_neigh=3, dist_type='l2'):
        n = len(self.train_images)
        values = np.zeros(n)
        if dist_type == 'l1':
            values = np.sum(abs(self.train_images - test_images), axis=1)  # calculam distanta Manhattan pe fiecare linie
        elif dist_type == 'l2':
            values = np.sqrt(np.sum((self.train_images - test_images) ** 2, axis=1))  # calculam distanta euclidiana pe fiecare linie

        classes_index = values.argsort()[:num_neigh]  # se vor sorta distantele si returna indicii corespunzatori, apoi ii luam pe primii num_heigh cei mai apropiati
        classes = self.train_classes[classes_index]  # preluam valorile corespunzatoare
        prediction = np.bincount(classes).argmax()  # luam clasa cu frecvenat cea mai mare
        return prediction


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
#------------------ validation images
validation_name_images = np.loadtxt('validation.txt', 'str', delimiter=',', usecols=0)
validation_classes = np.loadtxt('validation.txt', 'int', delimiter=',', usecols=1)

validation_images = []
for name in validation_name_images:
    img = image.imread('validation/' + name)
    arr = np.asarray(img)
    arr = np.reshape(arr, (1, 1024))      # size ul unei imagini este 32x32
    validation_images.append(arr[0])

validation_images = np.array(validation_images)

# ------------------ test images
test_name_images = np.loadtxt('test.txt', 'str', ) #citirea din fisier
test_images = []
for name in test_name_images:
    img = image.imread('test/' + name)
    arr = np.asarray(img)
    arr = np.reshape(arr, (1, 1024))
    test_images.append(arr[0])

test_images = np.array(test_images)



#ne cream clasificatorul
classifier = KnnClassifier(train_images, train_classes)
neighs= [9]  # am obtinut cea mai buna acuratete

# neighs = [3,5,7,9] am testat pentru mai multi vecini si am ajuns la concluzia ca n=9 este cel mai bun

nr_images = len(validation_images)
predictions = np.zeros(nr_images) #array in care vom pune predictiile
accuracy_l1 = np.zeros(len(neighs)) # array in care vom pune acuratetile, in cazul in care testam pentru mai multi vecini
j = 0
for n in neighs:
    for i in range(nr_images):
        predictions[i] = classifier.classify(validation_images[i], n, 'l1') # calculam predictiile pentru datele de validare pentru a calcula
    accuracy_l1[j] = accuracy_score(validation_classes, predictions)        # acurateta si matricea de confuzie
    print("L1:Accuracy score " + str(n) + " n: " + str(accuracy_l1[j])) # am calculat acuratetea
    print("Confusion matrix:")
    matrix = confusion_matrix(validation_classes, predictions) # am calculat matricea de confuzie
    print(matrix)
    j += 1
# accuracy_l2=[0.4760 , 0.4772, 0.483, 0.4916] obtinuta cu dist l2
nr_image = len(test_images)
neigh = 9
#am scris rezultatele in fisierul csv

with open('sample_submission2.txt', mode='w', newline='') as file: #punem newline= ' ' pentru a nu mai afisa un rand nou gol dupa fiecare linie
    file_wr = csv.writer(file, delimiter=',')  # datele vor fi delimitate de virgula in fisier, aflandu-se pe doua coloan
    file_wr.writerow(['id', 'label'])
    for i in range(nr_image): #parcurgem imaginile si le clasificam
        prediction = classifier.classify(test_images[i], neigh, 'l1')
        file_wr.writerow([test_name_images[i], prediction]) # apoi scriem in fisier rezultatul obtinut
