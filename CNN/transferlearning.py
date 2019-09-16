import numpy as np
import cv2

from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten,GlobalAveragePooling2D
from keras.models import Sequential
from keras import *
from keras import optimizers
from keras import losses
from keras.applications.vgg19 import VGG19
from keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

girisverisi=np.load("girisverisi.npy")
girisverisi=girisverisi.reshape(-1,224,224,3)/255  ##255 ile böldm
cikisverisi=np.array([ [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])

#validation_data için split
splitgiris=girisverisi[1:6]
splitgiris=np.append(splitgiris,girisverisi[24:29]).reshape(-1,224,224,3)/255  # 255 ile böl
splitcikis=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]])

vggmodel=VGG19(weights="imagenet",include_top=False) #include_top ? Flatten layeri alma diyoruzz
x=vggmodel.output
x=GlobalAveragePooling2D()(x) #tüm ağlarımı tek boyuta koyar.
x=Dropout(0.2)(x)
x=Dense(1024,activation="relu")(x)
x=Dropout(0.2)(x)
x=Dense(1024,activation="relu")(x)
x=Dropout(0.2)(x)
tahmin=Dense(2,activation='softmax')(x)


model=Model(inputs=vggmodel.input,outputs=tahmin)

for katman in vggmodel.layers:    ##vgg19 modelindeki katmanları tekrar güncelleme 
    katman.trainable=False

model.compile(optimizer=Adam(lr=0.0005),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()
model.fit(girisverisi,cikisverisi,epochs=10,batch_size=32,validation_data=(splitgiris,splitcikis))
model.save("Vgg19TransferLearning")
