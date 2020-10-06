import numpy as np
import tensorflow as tf
from PGD_attack import pgd
import sys
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization
mnist = tf.keras.datasets.mnist
(_,_),(x_test,_) = mnist.load_data()
x_test = np.reshape(x_test,(-1,28,28,1))


npzfile=np.load('data_prepare.npz')

for i in range(len(npzfile['X'])):
  v=int(npzfile['X'][i])
  img = pgd(x_test[v],v)
  img=(np.asarray(img) / 255.0).astype(np.float32)
  img=np.reshape(img,(1,28,28,1))
  if(i==0):
    t=img
  else:
    t=np.vstack((t,img))
  # print(np.shape(t))
x=t
y=npzfile['Y']



model = Sequential()
model.add(Conv2D (16, (3, 3),activation='relu', kernel_initializer='he_uniform', padding='same',input_shape= (28,28,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(GlobalMaxPooling2D())# compile model

opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

model.fit(x, y,
              batch_size=128,
              nb_epoch=64,
              shuffle=True)
model.save('train_cnn.h5')
