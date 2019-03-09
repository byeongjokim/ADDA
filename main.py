import pickle as pkl
import tensorflow as tf
import numpy as np

def get_dataset(istest=0):
    mnist = tf.keras.datasets.mnist
    (mnist_train_X, mnist_train_Y), (mnist_test_X, mnist_test_Y) = mnist.load_data()

    mnist_m = pkl.load(open("_data/mnist_m/mnistm_data.pkl", 'rb'))
    mnist_m_train_X = mnist_m["train"]
    mnist_m_test_X = mnist_m["test"]

    if(istest == 0):
        mnist_train_X = np.pad(mnist_train_X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        mnist_m_train_X = np.pad(mnist_m_train_X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        mnist_train_X = np.stack([mnist_train_X, mnist_train_X, mnist_train_X], 3)
        return mnist_train_X, mnist_train_Y, mnist_m_train_X
    else:
        mnist_test_X = np.pad(mnist_test_X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        mnist_m_test_X = np.pad(mnist_m_test_X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        mnist_test_X = np.stack([mnist_test_X, mnist_test_X, mnist_test_X], 3)
        return mnist_test_X, mnist_test_Y, mnist_m_test_X