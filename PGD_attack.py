import numpy as np
import os
import sys
import keras

from cleverhans.attacks import MadryEtAl
from PIL import Image
import imageio
from keras.models import load_model
import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper


sess = tf.keras.backend.get_session()
model=tf.keras.models.load_model('train.h5')
def pgd(img,i):
    batch_shape = [1, 28, 28, 1]
    epsilon=[0.03,0.06,0.1,0.2,0.3,0.4,0.7]
    tf.logging.set_verbosity(tf.logging.INFO)
    x = tf.placeholder(tf.float32, shape=batch_shape)
    eps=tf.placeholder(tf.float32,shape=())
    pgd1 = MadryEtAl(model, sess=sess)
    x_adv = pgd1.generate(x, eps=eps,clip_min=-1., clip_max=1.)
    img = (np.asarray(img) / 255.0).astype(np.float32)
    img = np.reshape(img,(1,28,28,1))
    adv_image = sess.run(x_adv, feed_dict={x: img,eps:epsilon[i%7]})
    adv_image = np.reshape(adv_image, (28, 28, 1))
    return img
