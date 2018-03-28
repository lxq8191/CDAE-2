import tensorflow as tf
from tqdm import trange
import numpy as np
from utils import *


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
        self.log = {'train_loss': [], 'ap@5': []}
        self.name = name

        self._build_model()

    def _build_model(self):
        hidden_units = self.hidden_units
        user_num = self.user_num
        item_num = self.item_num

        with tf.variable_scope(self.name):

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
                            -tf.reduce_sum((-self.targets*tf.log(self.decode) - \
                                    (1-self.targets)*tf.log(1-self.decode))*self.weights,
                                           reduction_indices=1))
                else:  # ===========================================================
                    self.cost = tf.reduce_mean(
                            tf.reduce_sum(-self.targets*tf.log(self.decode) - \
                                    (1-self.targets)*tf.log(1-self.decode),
                                    reduction_indices=1))

            else:
                raise NotImplementedError
            # ======================================================================

    def _init_optimizer(self):
        '''
        Initialize optimizer
        '''

        if self.optimizer == 'adadelta':
            self.optim = tf.train.AdadeltaOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'gradient':
            self.optim = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'adagrad':
            self.optim = tf.train.AdagradOptimizer(self.lr).minimize(self.cost)
        else:
            raise NotImplementedError

    def _run_optim(self, inputs, targets, user_id, weights=None):
        '''
        Run optimizer
        '''
        if weights is not None:
            self.optim.run(
                    session=self.sess,
                    feed_dict={
                        self.inputs: inputs,
                        self.targets: targets,
                        self.user_id: user_id,
                        self.weights: weights
                    })
        else:
            self.optim.run(
                    session=self.sess,
                    feed_dict={
                        self.inputs: inputs,
                        self.targets: targets,
                        self.user_id: user_id,
                    })
    def _get_loss(self, inputs, targets, user_id, weights=None):
        '''
        Get current loss
        '''
        if weights is not None:
            loss = self.cost.eval(
                    session=self.sess,
                    feed_dict={
                        self.inputs: inputs,
                        self.targets: targets,
                        self.user_id: user_id,
                        self.weights: weights
                    })
        else:
            loss = self.cost.eval(
                    session=self.sess,
                    feed_dict={
                        self.inputs: inputs,
                        self.targets: targets,
                        self.user_id: user_id
                    })
        return loss

    def train(self, rating, train_indices, test_indices, penalty_weights=None):

        self._init_optimizer()
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.with_weights and penalty_weights is None:
            raise ValueError

        for epoch in trange(self.epochs):
            loss_ = 0
            ap_at_5 = []
            for usr in range(self.user_num):
                input_ = [rating[usr]]
                target_ = [rating[usr]]

                if not self.with_weights:
                    self._run_optim(input_, target_, usr)
                    loss_ += self._get_loss(input_, target_, usr)
                else:
                    weights_ = [penalty_weights]
                    self._run_optim(input_, target_, usr, weights=weights_)
                    loss_ += self._get_loss(input_, target_, usr, weights=weights_)

                recon = self.recon.eval(
                    session=self.sess,
                    feed_dict={
                        self.inputs: [rating[usr]],
                        self.user_id: usr
                    })

                top5 = get_topN(recon, train_indices[usr])
                ap_at_5.append(avg_precision(top5, test_indices[usr]))
            self.log['train_loss'].append(loss_/self.user_num)
            self.log['ap@5'].append(sum(ap_at_5)/len(ap_at_5))


