from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
height_shift_range=0.1,shear_range=0.15, 
zoom_range=0.1,channel_shift_range = 10, horizontal_flip=True)

image_path="./145152.2.jpg"

image = np.expand_dims(plt.imread(image_path),axis=0)

os.mkdir("new_data")
save_here="./new_data/"

datagen.fit(image)

for x, val in zip(datagen.flow(image,                   
        save_to_dir=save_here,      
         save_prefix='aug',       
    
    save_format='png'),range(10)) :
	pass

