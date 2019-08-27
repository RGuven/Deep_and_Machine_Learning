import numpy as np
import cv2

from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras import *
from keras import optimizers
from keras import losses
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

girisverisi=np.load("girisverisi.npy")
girisverisi=girisverisi.reshape(-1,224,224,3)
cikisverisi=np.array([ [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])

#validation_data için split
splitgiris=girisverisi[1:6]
splitgiris=np.append(splitgiris,girisverisi[24:29]).reshape(-1,224,224,3)
splitcikis=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]])

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
model.summary()

model.fit(girisverisi/255,cikisverisi,epochs=3,batch_size=20,validation_data=(splitgiris,splitcikis)) #255 ile bölmek çokomelli

model.save("CnnAyakkabıTerlik")
model.load_weights("CnnAyakkabıTerlik")
print(model.get_weights()[0].shape)