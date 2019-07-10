import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd 

data=np.random.random((1000,1000))
labels=np.random.randint(2,size=(1000,1))
print("Data Shape =====>"+ str(data.shape))
print("Labels =========>"+ str((labels.shape)))

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=1000))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data,labels,epochs=130,batch_size=32)
predictions=model.predict(data)

model.summary()


dataframe_datas=pd.DataFrame(data)
print(dataframe_datas.head())
dataframe_labels=pd.DataFrame(labels)
print(dataframe_labels.info())

