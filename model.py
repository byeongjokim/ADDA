import tensorflow as tf
class Model(object):
    def __init__(self):
        print("a")
        self.mu = 0
        self.sigma = 0.1

        self.S = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.T = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        self.D = tf.placeholder(tf.float32, [None, 2])

    def lenet_source_CNN(self, image):
        with tf.variable_scope("source_cnn"):
            conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=self.mu, stddev=self.sigma))
            conv1_b = tf.Variable(tf.zeros(6))
            conv1 = tf.nn.conv2d(image, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
            conv1 = tf.nn.relu(conv1)
            pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=self.mu, stddev=self.sigma))
            conv2_b = tf.Variable(tf.zeros(16))
            conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
            conv2 = tf.nn.relu(conv2)
            pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            fc1 = tf.contrib.layers.flatten(pool_2)
            return fc1

    def lenet_target_CNN(self, image):
        with tf.variable_scope("target_cnn"):
            conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=self.mu, stddev=self.sigma))
            conv1_b = tf.Variable(tf.zeros(6))
            conv1 = tf.nn.conv2d(image, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
            conv1 = tf.nn.relu(conv1)
            pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=self.mu, stddev=self.sigma))
            conv2_b = tf.Variable(tf.zeros(16))
            conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
            conv2 = tf.nn.relu(conv2)
            pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            fc1 = tf.contrib.layers.flatten(pool_2)
            return fc1

    def lenet_classification(self, fc1):
        with tf.variable_scope("classification"):
            fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
            fc1_b = tf.Variable(tf.zeros(120))
            fc1 = tf.matmul(fc1, fc1_w) + fc1_b

            fc1 = tf.nn.relu(fc1)

            fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
            fc2_b = tf.Variable(tf.zeros(84))
            fc2 = tf.matmul(fc1, fc2_w) + fc2_b
            fc2 = tf.nn.relu(fc2)

            fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=self.mu, stddev=self.sigma))
            fc3_b = tf.Variable(tf.zeros(10))
            logits = tf.matmul(fc2, fc3_w) + fc3_b
            return logits

    def discriminate(self, fc1):
        with tf.variable_scope("discriminate", reuse=True):
            fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120)))
            fc1_b = tf.Variable(tf.zeros(120))
            fc1 = tf.matmul(fc1, fc1_w) + fc1_b

            fc1 = tf.nn.relu(fc1)

            fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84)))
            fc2_b = tf.Variable(tf.zeros(84))
            fc2 = tf.matmul(fc1, fc2_w) + fc2_b
            fc2 = tf.nn.relu(fc2)

            fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 2)))
            fc3_b = tf.Variable(tf.zeros(2))
            logits = tf.matmul(fc2, fc3_w) + fc3_b
            return logits

    def loss_function(self):
        source_M = self.lenet_source_CNN(self.S)
        target_M = self.lenet_target_CNN(self.T)
        cls = self.lenet_classification(source_M)
        D_source = self.discriminate(source_M)
        D_target = self.discriminate(target_M)

        self.cls_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls, labels=self.Y))
        self.dis_cost = - tf.reduce_mean(tf.log(D_source) + tf.log(1-D_target))
        self.target_M_cost = - tf.reduce_mean(tf.log(D_target))
