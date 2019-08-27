import numpy as np
import cv2

from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras import *
from keras import optimizers
from keras import losses
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model=Sequential()
model.add(Conv2D(50,11,strides=(4,4),input_shape=(224,224,3)))
model.add(MaxPooling2D(5,5))
model.add(Conv2D(50,5))
model.add(Conv2D(50,3))
model.add(Conv2D(50,3))
model.add(Conv2D(50,1))
model.add(Flatten())

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(4096,activation='relu'))
model.add(Dense(2))
model.add(Activation("softmax"))


model.compile(optimizer=optimizers.adam(lr=0.0001),loss="binary_crossentropy",metrics=["accuracy"])

model.load_weights("CnnAyakkabıTerlik")


girisverisi=np.array([])
 
uzantı="resimler/terlik1.jpg"
resim=cv2.imread(uzantı)
boyutlandır=cv2.resize(resim,(224,224))
girisverisi=np.append(girisverisi,boyutlandır)
girisverisi=girisverisi.reshape(-1,224,224,3)

print(np.round(model.predict(girisverisi))) 

girisverisi1=np.array([])
 
uzantı="resimler/ayak1.jpg"
resim=cv2.imread(uzantı)
boyutlandır=cv2.resize(resim,(224,224))
girisverisi1=np.append(girisverisi1,boyutlandır)
girisverisi1=girisverisi1.reshape(-1,224,224,3)

print(np.round(model.predict(girisverisi1))) 