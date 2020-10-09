from keras.preprocessing import image
from keras.models import load_model
import cv2
import numpy as np
import os
model = load_model("/vgg16_model.h5")
cam = cv2.VideoCapture(0)
while True:
  ret,frame = cam.read()
  height,width,c = frame.shape
  pic = frame
  pic = cv2.resize(pic,(224,224))
  predict = model.predict(np.array(pic).reshape(-1,224,224,3))
  predict = np.argmax(predict)
  gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier("/haarcascade.xml")
  faces = face_cascade.detectMultiScale(gray,1.2,2)
  if predict == 1:
    predict = "mask"
  else:
    predict = "no mask"
  for x,y,w,h in faces:
   color = (0,255,0) if predict == "mask" else (0,0,255)
   cv2.rectangle(pic,(x,y),(x+w,y+h),color,4)
   cv2.rectangle(pic,(x,y+h),(x+h,y+h+20),color,cv2.FILLED)
   cv2.putText(pic,predict,(x,y+h+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)    
   cv2.putText(pic,predict,(10,200),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
  pic = cv2.resize(pic,(450,400))
  cv2.imshow("test",pic)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    cv2.destroyAllWindows()
