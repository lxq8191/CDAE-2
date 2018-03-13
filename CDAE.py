import tensorflow as tf
from tqdm import trange
import numpy as np


class AutoEncoder(object):
    '''
    -- CDAE --
    Discription:
        Try to learn latent representation from rating matrix
    '''

    def __init__(
            self, sess, inputs, user_num, item_num, with_weights=False,
            targets=None, dropout_rate=0.2, lr=1., hidden_units=30, epochs=50,
            cost_function='rmse', name='autoencoder', b1=0.5, optimizer='adagrad'):
        '''
        -- Args --
            sess: tf session
            inputs: input matrix, shape = [1, item_num]
            user_num: number of users
            item_num: number of items
            targets: output targets, if None use conventional AutoEncoder,
                     otherwise, denoise AutoEncoder
            lr: learning rate
            hidden_units: number of middle layers units
            epochs: number of learning epoch
            b1: beta1, for adadelta optimizer
        '''

        self.sess = sess
        self.user_num = user_num
        self.item_num = item_num
        self.dropout_rate = dropout_rate
        self.b1 = b1
        self.lr = lr
        self.inputs = inputs
        self.epochs = epochs
        self.with_weights = with_weights
        self.hidden_units = hidden_units

        if targets is not None:
            self.targets = targets
        else:
            self.targets = inputs

        self.cost_function = cost_function
        self.optimizer = optimizer
        self.log = {'train_loss': [], 'valid_loss': []}
        self.name = name

        self._build_model()

    def _build_model(self):
        hidden_units = self.hidden_units
        user_num = self.user_num
        item_num = self.item_num

        with tf.variable_scope(self.name) as scope:

            ''' Generate user specifying vector '''
            # ======================================================================
            # initialize user specifying vector
            self.user_vector_matrix = tf.get_variable(
                    'autoencoder/user_vector',
                    shape=(user_num, hidden_units),
                    initializer=tf.random_normal_initializer(stddev=0.5),
                    dtype=tf.float32)

            # used for look up vector corresponding to user_id
            self.user_id = tf.placeholder(tf.int32, shape=[])

            # get user vector
            self.vec = tf.nn.embedding_lookup(self.user_vector_matrix, self.user_id)
            self.vec = tf.expand_dims(self.vec, 0)

            self.usrVec = tf.layers.dense(
                    inputs=self.vec,
                    units=hidden_units,
                    name='usrVec_dense')
            # ======================================================================

            ''' Denoise inputs '''
            # ======================================================================
            # denoising
            self.noise_inputs = tf.nn.dropout(self.inputs, 1-self.dropout_rate)
            # ======================================================================

            ''' AutoEncode '''
            # ======================================================================
            self.enc = tf.layers.dense(
                    inputs=self.noise_inputs,
                    units=hidden_units,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0003),
                    name='enc')

            self.code = tf.nn.sigmoid(tf.add(self.enc, self.usrVec))
            # self.code = tf.add(self.enc, self.usrVec)

            self.decode = tf.layers.dense(
                    inputs=self.code,
                    units=item_num,
                    activation=tf.nn.sigmoid,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0003),
                    name='dec')

            # self.recon_ = tf.round(self.decode)

            if self.cost_function == 'rmse':
                self.recon = self.decode
                self.cost = tf.sqrt(
                        tf.reduce_mean(tf.pow(self.targets-self.decode, 2)))
                self.cost_ = tf.sqrt(
                        tf.reduce_mean(tf.pow(self.targets-self.decode, 2)))
            
            elif self.cost_function == 'log_loss':
                self.recon = self.decode
                if self.with_weights:  # ===========================================
                    self.weights = tf.placeholder(tf.float32, shape=(1, self.item_num))
                    self.cost = tf.reduce_mean(
                            -tf.reduce_sum(self.targets*tf.log(self.decode)*self.weights,
                                           reduction_indices=1))
                else:  # ===========================================================
                    self.cost = tf.reduce_mean(
                            -tf.reduce_sum(self.targets*tf.log(self.decode),
                                           reduction_indices=1))

            else:
                raise NotImplementedError
            # ======================================================================

    def train(self, rating, penalty_weights=None):
        if self.optimizer == 'adadelta':
            self.optim = tf.train.AdadeltaOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'gradient':
            self.optim = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'adagrad':
            self.optim = tf.train.AdagradOptimizer(self.lr).minimize(self.cost)
        else:
            raise NotImplementedError

        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.with_weights:
            if penalty_weights is None:
                raise ValueError

        if self.targets is not self.inputs:
            if self.with_weights:  # ========================================
                for epoch in trange(self.epochs):
                    loss_ = 0
                    for usr in range(self.user_num):
                        self.optim.run(
                                session=self.sess,
                                feed_dict={
                                    self.inputs: [rating[usr]],
                                    self.targets: [rating[usr]],
                                    self.user_id: usr,
                                    self.weights: [penalty_weights]
                                })

                        loss_ += self.cost.eval(
                                session=self.sess,
                                feed_dict={
                                    self.inputs: [rating[usr]],
                                    self.targets: [rating[usr]],
                                    self.user_id: usr,
                                    self.weights: [penalty_weights]
                                })
                    self.log['train_loss'].append(loss_/self.user_num)
            else:  # ========================================================
                for epoch in trange(self.epochs):
                    loss_ = 0
                    for usr in range(self.user_num):
                        self.optim.run(
                                session=self.sess,
                                feed_dict={
                                    self.inputs: [rating[usr]],
                                    self.targets: [rating[usr]],
                                    self.user_id: usr,
                                })

                        loss_ += self.cost.eval(
                                session=self.sess,
                                feed_dict={
                                    self.inputs: [rating[usr]],
                                    self.targets: [rating[usr]],
                                    self.user_id: usr,
                                })
                    self.log['train_loss'].append(loss_/self.user_num)

    def averagePrecision(self, rating):
        '''
        Calculate Average Precision
        
        -- Args --:
            rating: rating matrix

        -- Return --:
            ap: list of average precision for each user
        '''
 
        ap = []
        for usr in range(rating.shape[0]):
            recon = self.recon.eval(
                    session=self.sess,
                    feed_dict={
                        self.inputs: [rating[usr]],
                        self.user_id: usr
                    })
            top5 = recon[0].argsort()[-5:][::-1]
            sum_p = 0.

            for k in range(5):
                count = 0
                if rating[usr][top5[k]] == 1:
                    for idx in range(k):
                        count += 1 if rating[usr][top5[idx]] == 1 else 0
                    sum_p += count / 5
            ap.append(sum_p / 5)

        return ap
