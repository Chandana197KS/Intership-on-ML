
from tensorflow.keras.models import Sequential #cnn
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Dropout,Flatten,Dense
import numpy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

images=numpy.load('images.npy')
labels=numpy.load('labels.npy')

model=Sequential()

#build model
model.add(Conv2D(256,kernel_size=(3,3),input_shape=images.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3)))
model.add(Activation('relu')) #rectified linear unit removes negetive pixels
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(60,activation="relu"))
model.add(Dense(2,activation="softmax")) #output layer

#compilation
model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.00001,decay=0.000001),metrics=['accuracy'])


#training
X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2)

ch=ModelCheckpoint("Maskdata.hdf5",verbose=0,monitor="val_loss",save_best_only=True,mode="auto")

model.fit(X_train,y_train,epochs=10,callbacks=[ch],validation_split=0.2)

print("training done....")