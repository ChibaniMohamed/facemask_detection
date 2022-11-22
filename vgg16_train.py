from keras.preprocessing import image
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from sklearn.utils import shuffle
import os
train_dir = "./train/"
test_dir = "./test/"
def data_loader(dir):
  labels = []
  pictures = []
  classe = 0
  for classes in os.listdir(dir):
    if classes == "with_mask":
     classe = 1
    else:
     classe = 0
    for pic in os.listdir(str(dir)+"/"+str(classes)+"/"):
      img = cv2.imread(str(dir)+"/"+str(classes)+"/"+pic)
      img = cv2.resize(img,(224,224))
      pictures.append(img)
      labels.append(classe)
  return shuffle(pictures,labels,random_state=10000)
x_train,y_train = data_loader(train_dir)
x_test,y_test = data_loader(test_dir)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

vgg = VGG16()
model = Sequential()
for layers in vgg.layers[:-1]:
  model.add(layers)
for l in model.layers:
  l.trainable = False
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
save = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5)

model.save("./vgg16_model.h5")
