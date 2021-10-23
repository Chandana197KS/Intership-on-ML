import cv2
from tensorflow.keras.models import load_model
import numpy

model=load_model('Maskdata.hdf5')

labels={1:"with mask",0:"without mask"}
cam=cv2.VideoCapture(0)
path="haarcascade_frontalface_default.xml"
df=cv2.CascadeClassifier(path)
while True:
    r,image=cam.read()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=df.detectMultiScale(gray,1.4,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
        face=gray[y:y+h,x:x+w]
        resized=cv2.resize(face,(120,120))
        normalizeimg=resized/255.0
        #reshapedimg=numpy.reshape(normalizeimg(1,120,120,1))
        reshapedimg=numpy.expand_dims(numpy.expand_dims(normalizeimg,-1),0)
        
        output=model.predict(reshapedimg)
        maxop=numpy.argmax(output,axis=1)[0]
        finalop=labels[maxop]
        
        cv2.putText(image,finalop,(100,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        
        cv2.imshow("Face",image)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
        
cam.release()
cv2.destroyAllWindows()