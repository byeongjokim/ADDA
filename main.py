import pickle as pkl
import tensorflow as tf
import numpy as np
from model import Model
import random

def get_dataset(istest=0):
    mnist = tf.keras.datasets.mnist
    (mnist_train_X, mnist_train_Y), (mnist_test_X, mnist_test_Y) = mnist.load_data()

    mnist_m = pkl.load(open("_data/mnist_m/mnistm_data.pkl", 'rb'))
    mnist_m_train_X = mnist_m["train"]
    mnist_m_test_X = mnist_m["test"]

    mnist_m_y = pkl.load(open("_data/mnist_m/mnistm_data_y.pkl", 'rb'))
    mnist_m_train_Y = mnist_m_y["train"]
    mnist_m_test_Y = mnist_m_y["test"]

    if(istest == 0):
        mnist_train_X = np.pad(mnist_train_X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        mnist_m_train_X = np.pad(mnist_m_train_X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")
        mnist_train_X = np.stack([mnist_train_X, mnist_train_X, mnist_train_X], 3)

        mnist_train_Y_one_hot = np.zeros((mnist_train_Y.shape[0], 10))
        mnist_train_Y_one_hot[np.arange(mnist_train_Y.shape[0]), mnist_train_Y] = 1

        mnist_m_train_Y_one_hot = np.zeros((mnist_m_train_Y.shape[0], 10))
        mnist_m_train_Y_one_hot[np.arange(mnist_m_train_Y.shape[0]), mnist_m_train_Y] = 1
        return [mnist_train_X, mnist_train_Y_one_hot], [mnist_m_train_X, mnist_m_train_Y_one_hot]
    else:
        mnist_test_X = np.pad(mnist_test_X, ((0, 0), (2, 2), (2, 2)), mode="constant")
        mnist_m_test_X = np.pad(mnist_m_test_X, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")
        mnist_test_X = np.stack([mnist_test_X, mnist_test_X, mnist_test_X], 3)

        mnist_test_Y_one_hot = np.zeros((mnist_test_Y.shape[0], 10))
        mnist_test_Y_one_hot[np.arange(mnist_test_Y.shape[0]), mnist_test_Y] = 1

        mnist_m_test_Y_one_hot = np.zeros((mnist_m_test_Y.shape[0], 10))
        mnist_m_test_Y_one_hot[np.arange(mnist_m_test_Y.shape[0]), mnist_m_test_Y] = 1
        return [mnist_test_X, mnist_test_Y_one_hot], [mnist_m_test_X, mnist_m_test_Y_one_hot]


def train_source():
    batch_size = 30
    learning_rate = 0.001
    epoch = 1000

    m = Model()
    loss, S, Y = m.loss_function(opt="pretrain")

    all_vars = tf.global_variables()
    var_source = [k for k in all_vars if k.name.startswith("source_cnn")]
    var_cls = [k for k in all_vars if k.name.startswith("classifier")]

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

    target, source = get_dataset(istest=0)
    source_X, source_Y = source

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_source = tf.train.Saver(var_source)
        saver_cls = tf.train.Saver(var_cls)
        saver_source.restore(sess, "./_models/source/source.ckpt")
        saver_cls.restore(sess, "./_models/classifier/classifier.ckpt")

        for e in range(epoch):
            total_cost = 0.0
            for i in range(0, source_X.shape[0], batch_size):
                _, cost = sess.run([optimizer, loss], feed_dict={
                                                    S: source_X[i:i+batch_size],
                                                    Y: source_Y[i:i+batch_size]
                                                  })
                total_cost = total_cost + cost

            if(total_cost / int(source_X.shape[0] / batch_size) < 0.05):
                break

            print(e, total_cost / int(source_X.shape[0] / batch_size))

        saver_source.save(sess, "./_models/source/source.ckpt")
        saver_cls.save(sess, "./_models/classifier/classifier.ckpt")

def train_adda():
    batch_size = 30
    learning_rate = 0.001
    epoch = 50

    m = Model()
    loss1, loss2, cls, S, T, Y = m.loss_function(opt="train")

    all_vars = tf.global_variables()
    var_source = [k for k in all_vars if k.name.startswith("source_cnn")]
    var_target = [k for k in all_vars if k.name.startswith("target_cnn")]
    var_dis = [k for k in all_vars if k.name.startswith("discriminater")]
    var_cls = [k for k in all_vars if k.name.startswith("classifier")]

    optimizer1 = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss1, var_list=var_dis)
    optimizer2 = tf.train.AdadeltaOptimizer(learning_rate=0.00001).minimize(loss2, var_list=var_target)

    is_correct = tf.equal(tf.argmax(cls, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(is_correct, tf.float32))

    target, source = get_dataset(istest=0)
    target_X, _ = target
    source_X, _ = source

    target_, _ = get_dataset(istest=1)
    test_X, test_Y = target_

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_source = tf.train.Saver(var_source)
        saver_source.restore(sess, "./_models/source/source.ckpt")
        saver_cls = tf.train.Saver(var_cls)
        saver_cls.restore(sess, "./_models/classifier/classifier.ckpt")
        saver_target = tf.train.Saver(var_target)
        saver_target.restore(sess, "./_models/target_pretrain/source.ckpt")
        saver_dis = tf.train.Saver(var_dis)

        saver_target.restore(sess, "./_models/target/target.ckpt")
        saver_dis.restore(sess, "./_models/discriminater/discriminater.ckpt")

        for e in range(epoch):
            total_cost1 = 0.0
            total_cost2 = 0.0
            for i in range(0, source_X.shape[0], batch_size):
                _, cost1 = sess.run([optimizer1, loss1], feed_dict={
                    S: source_X[i:i + batch_size],
                    T: target_X[i:i + batch_size]
                })

                _, cost2 = sess.run([optimizer2, loss2], feed_dict={
                    T: target_X[i:i + batch_size]
                })

                total_cost1 = total_cost1 + cost1
                total_cost2 = total_cost2 + cost2

            print(e, total_cost1 / int(source_X.shape[0] / batch_size), total_cost2 / int(source_X.shape[0] / batch_size))

            total_acc = 0.0
            for i in range(0, test_X.shape[0], batch_size):
                acc = sess.run(accuracy, feed_dict={
                    T: test_X[i:i + batch_size],
                    Y: test_Y[i:i + batch_size]
                })
                total_acc = total_acc + acc

            print(total_acc / test_X.shape[0])

            if(total_acc / test_X.shape[0] > 0.96):
                break

        saver_target.save(sess, "./_models/target/target.ckpt")
        saver_dis.save(sess, "./_models/discriminater/discriminater.ckpt")

def test_soruce():
    batch_size = 30

    m = Model()
    cls, S, Y = m.source()

    all_vars = tf.global_variables()
    var_source = [k for k in all_vars if k.name.startswith("source_cnn")]
    var_cls = [k for k in all_vars if k.name.startswith("classifier")]

    is_correct = tf.equal(tf.argmax(cls, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(is_correct, tf.float32))

    target, source = get_dataset(istest=1)
    target_X, target_Y = target
    #target_X, target_Y = source

    with tf.Session() as sess:
        saver_source = tf.train.Saver(var_source)
        saver_cls = tf.train.Saver(var_cls)
        saver_source.restore(sess, "./_models/source/source.ckpt")
        saver_cls.restore(sess, "./_models/classifier/classifier.ckpt")

        total_acc = 0.0
        for i in range(0, target_X.shape[0], batch_size):
            acc = sess.run(accuracy, feed_dict={
                S: target_X[i:i + batch_size],
                Y: target_Y[i:i + batch_size]
            })
            total_acc = total_acc + acc

        print(total_acc / target_X.shape[0])

def test_adda():
    batch_size = 30

    m = Model()
    cls, T, Y = m.adda()

    all_vars = tf.global_variables()
    var_target = [k for k in all_vars if k.name.startswith("target_cnn")]
    var_cls = [k for k in all_vars if k.name.startswith("classifier")]

    is_correct = tf.equal(tf.argmax(cls, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(is_correct, tf.float32))

    target, source = get_dataset(istest=1)
    target_X, target_Y = target

    with tf.Session() as sess:
        saver_target = tf.train.Saver(var_target)
        saver_cls = tf.train.Saver(var_cls)
        saver_target.restore(sess, "./_models/target/target.ckpt")
        saver_cls.restore(sess, "./_models/classifier/classifier.ckpt")

        total_acc = 0.0
        for i in range(0, target_X.shape[0], batch_size):
            acc = sess.run(accuracy, feed_dict={
                T: target_X[i:i + batch_size],
                Y: target_Y[i:i + batch_size]
            })
            total_acc = total_acc + acc

        print(total_acc / target_X.shape[0])

#train_source()
#test_soruce()
train_adda()
#test_adda()