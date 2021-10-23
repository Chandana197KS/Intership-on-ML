import cv2
import os
import numpy
from tensorflow.keras.utils import to_categorical

folders=['dataset/with mask','dataset/without mask']
lables={'dataset/with mask':0,'dataset/without mask':1}

images=[]
lablesfinal=[]

for folder in folders:
    imgs=os.listdir(folder)
    for img in imgs:
        imagepath=os.path.join(folder,img) #dataset/with mask/0-with-mask.jpg
        image=cv2.imread(imagepath)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resizeimg=cv2.resize(gray,(120,120))
        
        images.append(resizeimg)
        lablesfinal.append(lables[folder])
        
images=numpy.array(images)/255.0
images=numpy.reshape(images,(images.shape[0],120,120,1))
lablesfinal=numpy.array(lablesfinal)

newlabels=to_categorical(lablesfinal)

numpy.save("images",images)
numpy.save("labels",newlabels)

print("dataset conversion done..")