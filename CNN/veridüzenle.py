import numpy as np 
import cv2 


girisverisi=np.array([])
for i in range(30):
    i=i+1
    uzantı="resimler/%s.jpg"%i
    resim=cv2.imread(uzantı)
    boyutlandır=cv2.resize(resim,(224,224))
    girisverisi=np.append(girisverisi,boyutlandır)
    print(i)

print(girisverisi.shape)
girisverisi=girisverisi.reshape(30,224,224,3)  #30x224x224 
print(girisverisi.shape)
np.save("girisverisi",girisverisi)