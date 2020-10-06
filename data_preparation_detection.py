import skimage.transform
import PIL
import scipy.misc
import skimage
import imageio
from MSCN import calculate_brisque_features
import tensorflow as tf
from PGD_attack import pgd

mnist = tf.keras.datasets.mnist
(_,_),(x_test,_) = mnist.load_data()

x_test = np.reshape(x_test,(-1,28,28,1))

for i in range(1000):

        img = x_test[i]
        img=pgd(img,i)
        parameters=calculate_brisque_features(img, kernel_size=7, sigma=7/6)
        t = np.hstack((1, parameters))
        if i == 0:
            v = t
        else:
            v = np.vstack((v, t))


        img = x_test[i]
        parameters=calculate_brisque_features(img, kernel_size=7, sigma=7/6)
        t = np.hstack((0, parameters))
        v = np.vstack((v, t))


print(np.shape(v))
np.savez_compressed('data_training', X=v[:, 1:], Y=v[:, 0: 1])
