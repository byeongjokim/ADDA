import tensorflow as tf
class Model(object):
    def __init__(self):
        self.mu = 0
        self.sigma = 0.1

        self.S = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.T = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        self.D = tf.placeholder(tf.float32, [None, 2])

    def lenet_source_CNN(self, image):
        with tf.variable_scope("source_cnn"):
            conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 6], mean=self.mu, stddev=self.sigma))
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

            fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
            fc1_b = tf.Variable(tf.zeros(120))
            fc1 = tf.matmul(fc1, fc1_w) + fc1_b
            fc1 = tf.nn.relu(fc1)

            fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
            fc2_b = tf.Variable(tf.zeros(84))
            fc2 = tf.matmul(fc1, fc2_w) + fc2_b
            fc2 = tf.nn.tanh(fc2)
            return fc2

    def lenet_target_CNN(self, image):
        with tf.variable_scope("target_cnn"):
            conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 6], mean=self.mu, stddev=self.sigma))
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

            fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
            fc1_b = tf.Variable(tf.zeros(120))
            fc1 = tf.matmul(fc1, fc1_w) + fc1_b

            fc1 = tf.nn.relu(fc1)

            fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
            fc2_b = tf.Variable(tf.zeros(84))
            fc2 = tf.matmul(fc1, fc2_w) + fc2_b
            fc2 = tf.nn.tanh(fc2)
            return fc2

    def lenet_classification(self, fc2):
        with tf.variable_scope("classifier"):
            fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=self.mu, stddev=self.sigma))
            fc3_b = tf.Variable(tf.zeros(10))
            logits = tf.matmul(fc2, fc3_w) + fc3_b
            return logits

    def discriminate(self, fc2, reuse=None):
        with tf.variable_scope("discriminater") as scope:
            if reuse:
                scope.reuse_variables()
            fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 128), mean=self.mu, stddev=self.sigma))
            fc3_b = tf.Variable(tf.zeros(128))
            fc3 = tf.matmul(fc2, fc3_w) + fc3_b
            fc3 = tf.nn.relu(fc3)

            fc4_w = tf.Variable(tf.truncated_normal(shape=(128, 128), mean=self.mu, stddev=self.sigma))
            fc4_b = tf.Variable(tf.zeros(128))
            fc4 = tf.matmul(fc3, fc4_w) + fc4_b
            fc4 = tf.nn.relu(fc4)

            fc5_w = tf.Variable(tf.truncated_normal(shape=(128, 1), mean=self.mu, stddev=self.sigma))
            fc5_b = tf.Variable(tf.zeros(1))
            logits = tf.matmul(fc4, fc5_w) + fc5_b

            return logits

    def loss_function(self, opt):
        if(opt == "pretrain"):
            source_M = self.lenet_source_CNN(self.S)
            cls = self.lenet_classification(source_M)
            #self.cls_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cls, labels=self.Y))
            self.cls_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls, labels=self.Y))
            return self.cls_cost, self.S, self.Y

        if(opt == "train"):
            source_M = self.lenet_source_CNN(self.S)
            target_M = self.lenet_target_CNN(self.T)

            D_source_logits = self.discriminate(source_M)
            D_target_logits = self.discriminate(target_M, reuse=True)

            d_loss_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_source_logits,
                                                                                 labels=tf.ones_like(
                                                                                     D_source_logits)))  # log(D(x))
            d_loss_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_target_logits,
                                                                                 labels=tf.zeros_like(
                                                                                     D_target_logits)))  # log(1-D(G(z)))

            self.dis_cost = d_loss_source + d_loss_target  # log(D(x)) + log(1-D(G(z)))
            self.target_M_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_target_logits, labels=tf.ones_like(D_target_logits))) #log(D(G(z)))

            self.cls = self.lenet_classification(target_M)

            return self.dis_cost, self.target_M_cost, self.cls, self.S, self.T, self.Y

    def source(self):
        source_M = self.lenet_source_CNN(self.S)
        cls = self.lenet_classification(source_M)
        return cls, self.S, self.Y

    def adda(self):
        target_M = self.lenet_target_CNN(self.T)
        cls = self.lenet_classification(target_M)
        return cls, self.T, self.Y


