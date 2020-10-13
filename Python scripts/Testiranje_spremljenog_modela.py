import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import tensorflow as tf
import time
import pandas as pd
from sklearn import metrics
from keras.preprocessing import image
from keras import models
import matplotlib.image as mpimg

from PIL import Image

# fname = '1.jpg'
# image = Image.open(fname).convert("L")
# arr = np.asarray(image)
# # new_array = cv2.resize(arr, (25,25))
# # plt.imshow(new_array, cmap='gray', vmin=0, vmax=255)
# # plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
# plt.show()
# # img=cv2.imread("1.jpg")
# # cv2.imshow("Slika-Grayscale",img)
# # new_array = cv2.resize(img, (50,50))
# # cv2.imshow("smanjena dimenzija slike",new_array)
# # # cv2.imwrite("primjer grayscale slike.jpg",img)
# # cv2.waitKey(0)

# DATADIR = "C:/Users/Kresimir Markota/Desktop/Diplomski rad/Provjera"
# CATEGORIES = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]
#
# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#
#         break
#     break
#
# IMG_SIZE = 100
# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#
# def prepare(filepath):
#     IMG_SIZE=100
#     img_array_p=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
#     img_array_p=img_array_p/255.0
#     new_array_p=cv2.resize(img_array_p,(IMG_SIZE,IMG_SIZE))
#     return new_array_p.reshape(-1,IMG_SIZE,IMG_SIZE,1)
#
# prediction_data = []
# IMG_SIZE=100
# def create_prediction_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category)
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 prediction_data.append([new_array, class_num])
#             except Exception as e:
#                 pass
#
#
# create_prediction_data()
#
#
# random.shuffle(prediction_data)
#
# X_prediction = []
# y_prediction = []
#
# for features, label in prediction_data:
#     X_prediction.append(features)
#     y_prediction.append(label)
#
#
# X_prediction = np.array(X_prediction).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# X_prediction= X_prediction/255.0
#
#
# X = pickle.load(open("X.pickle","rb"))
# y = pickle.load(open("y.pickle","rb"))
#
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y = np.array(y)
#
# print(X_prediction.shape)
# print("---------------------")
# print(prepare("1.jpg").shape)
#
# model=tf.keras.models.load_model("bez_dropouta-1-conv-128-nodes-0-dense-1600868704-my_model.model")
#
#
# predict=model.predict_classes(X_prediction)
# predict=np.array(predict)
# print(predict)
#
# print(y)
# res=tf.math.confusion_matrix(y_prediction,predict)
# print("Confusion Matrix : ",res)


# img=image.load_img("1.jpg",grayscale=True,color_mode='rgb',target_size=(100,100))
# # img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img, axis = 0)
# img_tensor = img_tensor / 255.0
# print(img_tensor.shape)
# img_array_p=cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)
# img_array_p=img_array_p/255.0
# new_array_p=cv2.resize(img_array_p,(IMG_SIZE,IMG_SIZE))
#
# plt.imshow(new_array_p)
# # plt.show()
# new_array_p = np.array(new_array_p).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# layer_names = []
#
# for layer in model.layers[1:]:
#     layer_names.append(layer.name)
# print(layer_names)
# print(model.input)
# # Outputs of the 8 layers, which include conv2D and max pooling layers
# layer_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(new_array_p)
#
# # Getting Activations of first layer
# first_layer_activation = activations[0]
#
# # shape of first layer activation
# print(first_layer_activation.shape)
#
# # 6th channel of the image after first layer of convolution is applied
# plt.matshow(first_layer_activation[0, :, :, 1])
#
# # 15th channel of the image after first layer of convolution is applied
# plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
#
# plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
#
# plt.matshow(first_layer_activation[0, :, :, 6], cmap='viridis')
# plt.matshow(first_layer_activation[0, :, :, 16], cmap='viridis')
# plt.matshow(first_layer_activation[0, :, :, 17], cmap='viridis')
# plt.show()
#
#
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import tensorflow as tf
import time
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
x1 = pickle.load(open("trainHistoryDict-1-conv-32-nodes-0-dense-1600883786.pickle","rb"))
x2 = pickle.load(open("trainHistoryDict-1-conv-32-nodes-1-dense-1600889844.pickle","rb"))
x3= pickle.load(open("trainHistoryDict-1-conv-32-nodes-2-dense-1600896154.pickle","rb"))
x4= pickle.load(open("trainHistoryDict-1-conv-64-nodes-0-dense-1600884939.pickle","rb"))
x5= pickle.load(open("trainHistoryDict-1-conv-64-nodes-1-dense-1600891044.pickle","rb"))
x6= pickle.load(open("trainHistoryDict-1-conv-64-nodes-2-dense-1600897412.pickle","rb"))
x7= pickle.load(open("trainHistoryDict-1-conv-128-nodes-0-dense-1600887044.pickle","rb"))
x8= pickle.load(open("trainHistoryDict-1-conv-128-nodes-1-dense-1600893363.pickle","rb"))
x9= pickle.load(open("trainHistoryDict-1-conv-128-nodes-2-dense-1600899865.pickle","rb"))
x10= pickle.load(open("trainHistoryDict-2-conv-32-nodes-0-dense-1600884139.pickle","rb"))
x11= pickle.load(open("trainHistoryDict-2-conv-32-nodes-1-dense-1600890185.pickle","rb"))
x12= pickle.load(open("trainHistoryDict-2-conv-32-nodes-2-dense-1600896543.pickle","rb"))
x13= pickle.load(open("trainHistoryDict-2-conv-64-nodes-0-dense-1600885572.pickle","rb"))
x14= pickle.load(open("trainHistoryDict-2-conv-64-nodes-1-dense-1600891690.pickle","rb"))
x15= pickle.load(open("trainHistoryDict-2-conv-64-nodes-2-dense-1600898150.pickle","rb"))
x16= pickle.load(open("trainHistoryDict-2-conv-128-nodes-0-dense-1600888249.pickle","rb"))
x17= pickle.load(open("trainHistoryDict-2-conv-128-nodes-1-dense-1600894588.pickle","rb"))
x18= pickle.load(open("trainHistoryDict-2-conv-128-nodes-2-dense-1600901113.pickle","rb"))
x19= pickle.load(open("trainHistoryDict-3-conv-32-nodes-0-dense-1600884502.pickle","rb"))
x20= pickle.load(open("trainHistoryDict-3-conv-32-nodes-1-dense-1600890556.pickle","rb"))
x21= pickle.load(open("trainHistoryDict-3-conv-32-nodes-2-dense-1600896913.pickle","rb"))
x22= pickle.load(open("trainHistoryDict-3-conv-64-nodes-0-dense-1600886256.pickle","rb"))
x23= pickle.load(open("trainHistoryDict-3-conv-64-nodes-1-dense-1600892372.pickle","rb"))
x24= pickle.load(open("trainHistoryDict-3-conv-64-nodes-2-dense-1600898862.pickle","rb"))
x25= pickle.load(open("trainHistoryDict-3-conv-128-nodes-0-dense-1600889572.pickle","rb"))
x26= pickle.load(open("trainHistoryDict-3-conv-128-nodes-1-dense-1600895878.pickle","rb"))
x27= pickle.load(open("trainHistoryDict-3-conv-128-nodes-2-dense-1600902421.pickle","rb"))

epochs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
print(len(x11['accuracy']))

plt.plot(epochs, x1['accuracy'], label='x1 Training accuracy ')
plt.plot(epochs, x2['accuracy'], label='x2 Training accuracy ')
plt.plot(epochs, x3['accuracy'], label='x3 Training accuracy ')
plt.plot(epochs, x4['accuracy'], label='x4 Training accuracy ')
plt.plot(epochs, x5['accuracy'], label='x5 Training accuracy ')
plt.plot(epochs, x6['accuracy'], label='x6 Training accuracy ')
plt.plot(epochs, x7['accuracy'], label='x7 Training accuracy ')
plt.plot(epochs, x8['accuracy'], label='x8 Training accuracy ')
plt.plot(epochs, x9['accuracy'], label='x9 Training accuracy ')
plt.plot(epochs, x10['accuracy'], label='x10 Training accuracy ')
plt.plot(epochs, x11['accuracy'], label='x11 Training accuracy ')
plt.plot(epochs, x12['accuracy'], label='x12 Training accuracy ')
plt.plot(epochs, x13['accuracy'], label='x13 Training accuracy ')
plt.plot(epochs, x14['accuracy'], label='x14 Training accuracy ')
plt.plot(epochs, x15['accuracy'], label='x15 Training accuracy ')
plt.plot(epochs, x16['accuracy'], label='x16 Training accuracy ')
plt.plot(epochs, x17['accuracy'], label='x17 Training accuracy ')
plt.plot(epochs, x18['accuracy'], label='x18 Training accuracy ')
plt.plot(epochs, x19['accuracy'], label='x19 Training accuracy ')
plt.plot(epochs, x20['accuracy'], label='x20 Training accuracy ')
plt.plot(epochs, x21['accuracy'], label='x21 Training accuracy ')
plt.plot(epochs, x22['accuracy'], label='x22 Training accuracy ')
plt.plot(epochs, x23['accuracy'], label='x23 Training accuracy ')
plt.plot(epochs, x24['accuracy'], label='x24 Training accuracy ')
plt.plot(epochs, x25['accuracy'], label='x25 Training accuracy ')
plt.plot(epochs, x26['accuracy'], label='x26 Training accuracy ')
plt.plot(epochs, x27['accuracy'], label='x27 Training accuracy ')


plt.title('Training accuracy with dropout')
plt.legend()
# plt.savefig("Training accuracy with dropout.jpg")
plt.show()
