import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


DATADIR = "C:/Users/Kresimir Markota/Desktop/Diplomski rad/DataSet"
CATEGORIES = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

        break
    break

IMG_SIZE = 100
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()


random.shuffle(training_data)

X = []
y = []
dataGen=ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	                        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	                        horizontal_flip=True, fill_mode="nearest")

dataGenDrugi =ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,fill_mode="nearest")
dataGenTreci =ImageDataGenerator(horizontal_flip=True,fill_mode="nearest")
dataGenCetvrti = ImageDataGenerator(rotation_range=15, width_shift_range=0.15,
	                                height_shift_range=0.15, shear_range=0.15, zoom_range=0.25,
	                                horizontal_flip=False, fill_mode="nearest")
dataGenPeti=ImageDataGenerator(rotation_range=30, width_shift_range=0.05,
	                                height_shift_range=0.05, shear_range=0.25, zoom_range=0.1,
	                                horizontal_flip=True)
for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
it=dataGen.flow(X,y,1)
it2=dataGenDrugi.flow(X,y,1)
it3=dataGenTreci.flow(X,y,1)
it4=dataGenCetvrti.flow(X,y,1)
it5=dataGenPeti.flow(X,y,1)

X=np.append(X,it.x,axis=0)
y=np.append(y,it.y,axis=0)
X=np.append(X,it2.x,axis=0)
y=np.append(y,it2.y,axis=0)
X=np.append(X,it3.x,axis=0)
y=np.append(y,it3.y,axis=0)
X=np.append(X,it4.x,axis=0)
y=np.append(y,it4.y,axis=0)
X=np.append(X,it5.x,axis=0)
y=np.append(y,it5.y,axis=0)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X.shape)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


