#https://github.com/aleju/imgaug
#smoth tekniğide var

import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator #İMPORTATNANTNATNANTNAT
from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras import *
from keras import optimizers
from keras import losses
import keras
print(keras.__version__, keras.__file__)
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
model.load_weights("CnnAyakkabıTerlik")

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)################################################################################# HATA ALIYORUM

datagen.fit(girisverisi/255)
model.fit_generator(datagen.flow(girisverisi/255, cikisverisi,
                        batch_size=32),
                        epochs=3,
                        steps_per_epoch=len(girisverisi/255) / 32)
                        
                        
                        
                        
                        #validation_data=(splitgiris/255, splitcikis),
                        #workers=1)



model.save("CNNwithdataAug")
model.load_weights("CNNwithdataAug")

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

