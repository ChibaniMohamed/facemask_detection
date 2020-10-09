from keras.preprocessing import image
from keras.models import Sequential,load_model
from keras.layers import Dense,Conv2D,Flatten,BatchNormalization,MaxPool2D,Dropout
import cv2
import numpy as np
from sklearn.utils import shuffle
import os
train_dir = "/data/train"
test_dir = "/data/test"
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

model = Sequential()
model.add(Conv2D(124,(3,3),activation="relu",input_shape=(224,224,3)))
model.add(Conv2D(124,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
save = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50)

model.save("./model.h5")
