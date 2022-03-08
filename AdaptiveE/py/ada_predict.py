import pdb
import sys

import numpy as np
import tensorflow as tf

from const import NE, NR, TEST_SIZE, TRAIN_SIZE, VAL_SIZE
from losses import (BootstrappedSigmoidClassificationLoss, GCELoss,
                    symmetric_cross_entropy, symmetric_cross_entropy_stable)
from utils import Scope, scatter_update_tensor, set_gpu, write_graph

# build create static ops
# call create ops dependent on logic

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class AdaE(object):
    def __init__(
        self,
        NE,
        NR,
        E,
        C=32,
        v_dim=10,
        inp_dp=0.2,
        hid_dp=0.2,
        last_dp=0.3,
        order=2,
        prefix="",
        reuse=None,
    ):
        super().__init__()
        self.NE = NE
        self.NR = NR
        self.E = E
        self.C = C
        self.v_dim = v_dim
        self.inp_dp = inp_dp
        self.hid_dp = hid_dp
        self.last_dp = last_dp
        self.order = order
        self.prefix = prefix
        self.reuse = reuse

        self.path = f"{prefix}AdaE/"

    def embedding_lookup(self, e1, rel, pretrained_embeddings=None):
        # [NE, E] - lookup -> [bs, E] -> [bs, E, 1]
        with Scope("embedding_lookup", prefix=self.path, reuse=self.reuse):
            self.emb_e = tf.get_variable(
                'emb_e',
                shape=[self.NE, self.E],
                initializer=tf.glorot_normal_initializer())

            self.emb_rel = tf.get_variable(
                'emb_r',
                shape=[self.NR, self.E],
                initializer=tf.glorot_normal_initializer())

            if pretrained_embeddings is not None:
                e_emb_extra = tf.nn.embedding_lookup(
                    pretrained_embeddings, e1, name="e1_pretrained_embedding")
                e1_emb = tf.reshape(
                    tf.add(
                        e_emb_extra,
                        tf.nn.embedding_lookup(self.emb_e,
                                               e1,
                                               name="e1_embedding")),
                    [-1, self.E, 1])
            else:
                e1_emb = tf.reshape(
                    tf.nn.embedding_lookup(self.emb_e, e1,
                                           name="e1_embedding"),
                    [-1, self.E, 1])

            rel_emb = tf.reshape(
                tf.nn.embedding_lookup(self.emb_rel, rel,
                                       name="rel_embedding"), [-1, self.E, 1])
        return e1_emb, rel_emb

    def transe(self, triple, neg_triple):
        with Scope("transe", prefix=self.path, reuse=self.reuse):
            pos_h_e = triple[0]
            pos_r_e = triple[1]
            pos_t_e = triple[2]
            neg_h_e = neg_triple[0]
            neg_r_e = neg_triple[1]
            neg_t_e = neg_triple[2]

            pos_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_h_e,
                                             name="pos_h_e")
            pos_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             pos_r_e,
                                             name="pos_rel_e")
            pos_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_t_e,
                                             name="pos_t_e")
            neg_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_h_e,
                                             name="neg_h_e")
            neg_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             neg_r_e,
                                             name="neg_rel_e")
            neg_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_t_e,
                                             name="neg_t_e")
            # l2 normalization
            pos_h_e = tf.nn.l2_normalize(pos_h_e, axis=1)
            pos_r_e = tf.nn.l2_normalize(pos_r_e, axis=1)
            pos_t_e = tf.nn.l2_normalize(pos_t_e, axis=1)
            neg_h_e = tf.nn.l2_normalize(neg_h_e, axis=1)
            neg_r_e = tf.nn.l2_normalize(neg_r_e, axis=1)
            neg_t_e = tf.nn.l2_normalize(neg_t_e, axis=1)

            dis_pos = tf.norm(pos_h_e + pos_r_e - pos_t_e, 2, keep_dims=True)
            dis_neg = tf.norm(neg_h_e + neg_r_e - neg_t_e, 2, keep_dims=True)
            return dis_pos, dis_neg

    def ic_emb(self, e1_emb, rel_emb, training):
        with Scope("ic_emb", prefix=self.path, reuse=self.reuse):
            self.bn_e1 = tf.layers.BatchNormalization(axis=-1, name="bn_e1")
            self.dp_e1 = tf.layers.Dropout(rate=self.inp_dp, name="dp_e1")

            self.bn_rel = tf.layers.BatchNormalization(axis=-1, name="bn_rel")
            self.dp_rel = tf.layers.Dropout(rate=self.inp_dp, name="dp_rel")

            e1_emb = self.dp_e1(self.bn_e1(e1_emb, training=training),
                                training=training)
            rel_emb = self.dp_rel(self.bn_rel(rel_emb, training=training),
                                  training=training)
        return e1_emb, rel_emb

    def make_ada_adj(self):
        with Scope("ada_vector", prefix=self.path, reuse=self.reuse) as scope:
            self.v1 = tf.get_variable(
                'v1',
                shape=[2 * self.E, self.v_dim],
                initializer=tf.glorot_normal_initializer())

            self.v2 = tf.get_variable(
                'v2',
                shape=[self.v_dim, 2 * self.E],
                initializer=tf.glorot_normal_initializer())

            A = tf.nn.softmax(tf.nn.relu(tf.matmul(self.v1, self.v2)),
                              axis=1,
                              name="ada_adj")
        return A

    def e1_rel_gcn(self, e1_emb, rel_emb, A):
        self.gcn = GCN(self.order, self.C, prefix=self.path)
        x = self.gcn(e1_emb, rel_emb, A, self.reuse)
        return x

    def ic_hid(self, x, training):
        with Scope("ic_hid", prefix=self.path, reuse=self.reuse):
            self.dp_hid = tf.layers.Dropout(rate=self.hid_dp)
            self.bn_hid = tf.layers.BatchNormalization(axis=-1)
            x = self.dp_hid(self.bn_hid(x, training=training),
                            training=training)
        return x

    def fc(self, x):
        # 2*E*C -> E
        return tf.layers.dense(
            x,
            self.E,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="fc")

    def ic_last(self, x, training):
        with Scope("ic_last", prefix=self.path, reuse=self.reuse):
            self.bn_last = tf.layers.BatchNormalization(axis=-1)
            self.dp_last = tf.layers.Dropout(rate=self.last_dp)
            x = self.dp_last(self.bn_last(x, training=training),
                             training=training)
        return x

    def __call__(self,
                 e1,
                 rel,
                 training=False,
                 pretrained_embeddings=None,
                 triple=None,
                 neg_triple=None):
        with Scope("AdaE", prefix=self.prefix, reuse=self.reuse) as scope:

            e1_emb, rel_emb = self.embedding_lookup(e1, rel,
                                                    pretrained_embeddings)

            if triple is not None:
                dis_pos, dis_neg = self.transe(triple, neg_triple)
            else:
                dis_pos = 0
                dis_neg = 1.

            # e1_emb, rel_emb = self.ic_emb(e1_emb, rel_emb, training)
            A = self.make_ada_adj()
            x = self.e1_rel_gcn(e1_emb, rel_emb, A)

            x = tf.nn.relu(x)
            # x = self.ic_hid(x, training)
            x = tf.layers.flatten(x, name="flatten")  # b, 2E, C -> b,2E*C

            x = tf.nn.relu(x)
            x = self.fc(x)

            # x = self.ic_last(x, training)
            # bs E  E, N -> bs, N
            with Scope("pred", prefix=scope.prefix,
                       reuse=self.reuse) as scope_pred:
                x = tf.matmul(x, self.emb_e, transpose_b=True, name="mul_E")
                self.b = tf.get_variable('bias',
                                         shape=[self.NE],
                                         initializer=tf.zeros_initializer())
                x = tf.add(x, self.b, name="logits")
        return x, dis_pos, dis_neg


class GCN(object):
    def __init__(self, order=2, C=32, prefix=""):
        super().__init__()
        self.order = order
        self.C = C
        self.prefix = prefix

    def __call__(self, x_e, x_rel, A, reuse=None):
        '''
        A: n x n
        '''
        with Scope("GCN", prefix=self.prefix, reuse=reuse) as scope:
            # E = tf.shape(x_e)[1]
            xs = []

            with tf.variable_scope(f"layer_0", reuse=reuse):
                x = tf.concat([x_e, x_rel], 1)

            for i in range(self.order):
                with Scope(f"layer_{i+1}", prefix=scope.prefix, reuse=reuse):
                    # nxn b n e -> b n e
                    x = tf.matmul(A, x)
                    x = tf.layers.dense(
                        x,
                        self.C,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        name=f"W{i+1}")

                    xs.append(x)
                    # FIXME tf.cond?
                    if i < self.order - 1:
                        x = tf.nn.relu(x)
            with Scope("readout", prefix=scope.prefix, reuse=reuse):
                h = tf.concat(xs, -1)
                h = tf.layers.dense(
                    h,
                    self.C,
                    name="fc_readout",
                    activation=None,
                    use_bias=True,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.zeros_initializer())
        return h


class AdaE2(object):
    def __init__(
        self,
        NE,
        NR,
        E,
        C=32,
        v_dim=10,
        inp_dp=0.2,
        hid_dp=0.2,
        last_dp=0.3,
        order=2,
        wd=0.0003,
        prefix="",
        reuse=None,
    ):
        super().__init__()
        self.NE = NE
        self.NR = NR
        self.E = E
        self.C = C
        self.v_dim = v_dim
        self.inp_dp = inp_dp
        self.hid_dp = hid_dp
        self.last_dp = last_dp
        self.order = order
        self.wd = wd
        self.prefix = prefix
        self.reuse = reuse

        self.path = f"{prefix}AdaE2/"

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=wd)

    def embedding_lookup(self, e1, rel, pretrained_embeddings=None):
        # [NE, E] - lookup -> [bs, E] -> [bs, E, 1]
        with Scope("embedding_lookup", prefix=self.path, reuse=self.reuse):
            self.emb_e = tf.get_variable(
                'emb_e',
                shape=[self.NE, self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            self.emb_rel = tf.get_variable(
                'emb_r',
                shape=[self.NR, self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            if pretrained_embeddings is not None:
                e_emb_extra = tf.nn.embedding_lookup(
                    pretrained_embeddings, e1, name="e1_pretrained_embedding")
                e1_emb = tf.reshape(
                    tf.add(
                        e_emb_extra,
                        tf.nn.embedding_lookup(self.emb_e,
                                               e1,
                                               name="e1_embedding")),
                    [-1, self.E, 1])
            else:
                e1_emb = tf.reshape(
                    tf.nn.embedding_lookup(self.emb_e, e1,
                                           name="e1_embedding"),
                    [-1, self.E, 1])

            rel_emb = tf.reshape(
                tf.nn.embedding_lookup(self.emb_rel, rel,
                                       name="rel_embedding"), [-1, self.E, 1])
        return e1_emb, rel_emb

    def transe(self, triple, neg_triple):
        with Scope("transe", prefix=self.path, reuse=self.reuse):
            pos_h_e = triple[0]
            pos_r_e = triple[1]
            pos_t_e = triple[2]
            neg_h_e = neg_triple[0]
            neg_r_e = neg_triple[1]
            neg_t_e = neg_triple[2]

            pos_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_h_e,
                                             name="pos_h_e")
            pos_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             pos_r_e,
                                             name="pos_rel_e")
            pos_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_t_e,
                                             name="pos_t_e")
            neg_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_h_e,
                                             name="neg_h_e")
            neg_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             neg_r_e,
                                             name="neg_rel_e")
            neg_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_t_e,
                                             name="neg_t_e")
            # l2 normalization
            pos_h_e = tf.nn.l2_normalize(pos_h_e, axis=1)
            pos_r_e = tf.nn.l2_normalize(pos_r_e, axis=1)
            pos_t_e = tf.nn.l2_normalize(pos_t_e, axis=1)
            neg_h_e = tf.nn.l2_normalize(neg_h_e, axis=1)
            neg_r_e = tf.nn.l2_normalize(neg_r_e, axis=1)
            neg_t_e = tf.nn.l2_normalize(neg_t_e, axis=1)

            dis_pos = tf.norm(pos_h_e + pos_r_e - pos_t_e, 2, keep_dims=True)
            dis_neg = tf.norm(neg_h_e + neg_r_e - neg_t_e, 2, keep_dims=True)
            return dis_pos, dis_neg

    def bn_emb(self, e1_emb, rel_emb, training):
        with Scope("bn_emb", prefix=self.path, reuse=self.reuse):
            self.bn_e1 = tf.layers.BatchNormalization(axis=-1, name="bn_e1")

            self.bn_rel = tf.layers.BatchNormalization(axis=-1, name="bn_rel")

            e1_emb = self.bn_e1(e1_emb, training=training)
            rel_emb = self.bn_rel(rel_emb, training=training)
        return e1_emb, rel_emb

    def make_ada_adj(self):
        with Scope("ada_vector", prefix=self.path, reuse=self.reuse) as scope:
            self.v1 = tf.get_variable(
                'v1',
                shape=[2 * self.E, self.v_dim],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            self.v2 = tf.get_variable(
                'v2',
                shape=[self.v_dim, 2 * self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            A = tf.nn.softmax(tf.nn.relu(tf.matmul(self.v1, self.v2)),
                              axis=1,
                              name="ada_adj")
        return A

    def e1_rel_gcn(self, e1_emb, rel_emb, A, training):
        self.gcn = GCN2(self.order, self.C, self.wd, prefix=self.path)
        x = self.gcn(e1_emb, rel_emb, A, training, self.reuse)
        return x

    def bn_hid(self, x, training):
        with Scope("bn_hid", prefix=self.path, reuse=self.reuse):
            self.bn_hid = tf.layers.BatchNormalization(axis=-1)
            x = self.bn_hid(x, training=training)
        return x

    def fc(self, x):
        # 2*E*C -> E
        return tf.layers.dense(
            x,
            self.E,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="fc")

    def bn_last(self, x, training):
        with Scope("bn_last", prefix=self.path, reuse=self.reuse):
            self.bn_last = tf.layers.BatchNormalization(axis=-1)
            x = self.bn_last(x, training=training)
        return x

    def __call__(self,
                 e1,
                 rel,
                 training=False,
                 pretrained_embeddings=None,
                 triple=None,
                 neg_triple=None):
        with Scope("AdaE2", prefix=self.prefix, reuse=self.reuse) as scope:
            e1_emb, rel_emb = self.embedding_lookup(e1, rel,
                                                    pretrained_embeddings)

            if triple is not None:
                dis_pos, dis_neg = self.transe(triple, neg_triple)
            else:
                dis_pos = 0
                dis_neg = 1.

            e1_emb, rel_emb = self.bn_emb(e1_emb, rel_emb, training)
            A = self.make_ada_adj()
            x = self.e1_rel_gcn(e1_emb, rel_emb, A, training)

            x = tf.layers.flatten(x, name="flatten")  # b, 2E, C -> b,2E*C
            x = self.bn_hid(x, training)
            x = tf.nn.relu(x)
            x = self.fc(x)
            x = self.bn_last(x, training)
            x = tf.nn.relu(x)
            # bs E  E, N -> bs, N
            with Scope("pred", prefix=scope.prefix,
                       reuse=self.reuse) as scope_pred:
                x = tf.matmul(x, self.emb_e, transpose_b=True, name="mul_E")
                self.b = tf.get_variable(
                    'bias',
                    shape=[self.NE],
                    initializer=tf.zeros_initializer(),
                    regularizer=self.regularizer,
                )
                x = tf.add(x, self.b, name="logits")
        return x, dis_pos, dis_neg


class GCN2(object):
    def __init__(
        self,
        order=2,
        C=32,
        wd=0.0003,
        prefix="",
    ):
        super().__init__()
        self.order = order
        self.C = C
        self.prefix = prefix

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=wd)

    def __call__(
        self,
        x_e,
        x_rel,
        A,
        training,
        reuse=None,
    ):
        '''
        A: n x n
        '''
        with Scope("GCN", prefix=self.prefix, reuse=reuse) as scope:
            # E = tf.shape(x_e)[1]
            xs = []

            with tf.variable_scope(f"layer_0", reuse=reuse):
                x = tf.concat([x_e, x_rel], 1)  #[None, 1024, 1]
                x_identity_0 = x

            for i in range(self.order):
                with Scope(f"layer_{i+1}", prefix=scope.prefix, reuse=reuse):
                    # nxn b n e -> b n e
                    x_identity = x
                    x = tf.matmul(A, x)
                    x = tf.layers.dense(
                        x,
                        self.C,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        kernel_regularizer=self.regularizer,
                        name=f"W{i+1}")
                    x = tf.add(x_identity, x)
                    xs.append(x)
                    # FIXME tf.cond?
                    if i < self.order - 1:
                        x = tf.layers.BatchNormalization(axis=-1)(
                            x, training=training)
                        x = tf.nn.relu(x)
            with Scope("readout", prefix=scope.prefix, reuse=reuse):
                h = tf.concat(xs, -1)  #[None, 1024, 32 * 2]
                h = tf.layers.BatchNormalization(axis=-1)(h, training=training)
                h = tf.relu(h)
                h = tf.layers.dense(
                    h,
                    self.C,
                    name="fc_readout",
                    activation=None,
                    use_bias=True,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=self.regularizer,
                )
                h = h + tf.layers.dense(
                    x_identity_0,
                    self.C,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    kernel_regularizer=self.regularizer)
        return h


class AdaE3(object):
    def __init__(
        self,
        NE,
        NR,
        E,
        C=32,
        v_dim=10,
        inp_dp=0.2,
        hid_dp=0.2,
        last_dp=0.3,
        order=2,
        wd=0.0003,
        prefix="",
        reuse=None,
    ):
        super().__init__()
        self.NE = NE
        self.NR = NR
        self.E = E
        self.C = C
        self.v_dim = v_dim
        self.inp_dp = inp_dp
        self.hid_dp = hid_dp
        self.last_dp = last_dp
        self.order = order
        self.wd = wd
        self.prefix = prefix
        self.reuse = reuse

        self.path = f"{prefix}AdaE3/"

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=wd)

    def embedding_lookup(self, e1, rel, pretrained_embeddings=None):
        # [NE, E] - lookup -> [bs, E] -> [bs, E, 1]
        with Scope("embedding_lookup", prefix=self.path, reuse=self.reuse):
            self.emb_e = tf.get_variable(
                'emb_e',
                shape=[self.NE, self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            self.emb_rel = tf.get_variable(
                'emb_r',
                shape=[self.NR, self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            if pretrained_embeddings is not None:
                e_emb_extra = tf.nn.embedding_lookup(
                    pretrained_embeddings, e1, name="e1_pretrained_embedding")
                e1_emb = tf.reshape(
                    tf.add(
                        e_emb_extra,
                        tf.nn.embedding_lookup(self.emb_e,
                                               e1,
                                               name="e1_embedding")),
                    [-1, self.E, 1])
            else:
                e1_emb = tf.reshape(
                    tf.nn.embedding_lookup(self.emb_e, e1,
                                           name="e1_embedding"),
                    [-1, self.E, 1])

            rel_emb = tf.reshape(
                tf.nn.embedding_lookup(self.emb_rel, rel,
                                       name="rel_embedding"), [-1, self.E, 1])
        return e1_emb, rel_emb

    def transe(self, triple, neg_triple):
        with Scope("transe", prefix=self.path, reuse=self.reuse):
            pos_h_e = triple[0]
            pos_r_e = triple[1]
            pos_t_e = triple[2]
            neg_h_e = neg_triple[0]
            neg_r_e = neg_triple[1]
            neg_t_e = neg_triple[2]

            pos_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_h_e,
                                             name="pos_h_e")
            pos_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             pos_r_e,
                                             name="pos_rel_e")
            pos_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_t_e,
                                             name="pos_t_e")
            neg_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_h_e,
                                             name="neg_h_e")
            neg_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             neg_r_e,
                                             name="neg_rel_e")
            neg_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_t_e,
                                             name="neg_t_e")
            # l2 normalization
            pos_h_e = tf.nn.l2_normalize(pos_h_e, axis=1)
            pos_r_e = tf.nn.l2_normalize(pos_r_e, axis=1)
            pos_t_e = tf.nn.l2_normalize(pos_t_e, axis=1)
            neg_h_e = tf.nn.l2_normalize(neg_h_e, axis=1)
            neg_r_e = tf.nn.l2_normalize(neg_r_e, axis=1)
            neg_t_e = tf.nn.l2_normalize(neg_t_e, axis=1)

            dis_pos = tf.norm(pos_h_e + pos_r_e - pos_t_e, 2, keep_dims=True)
            dis_neg = tf.norm(neg_h_e + neg_r_e - neg_t_e, 2, keep_dims=True)
            return dis_pos, dis_neg

    def make_ada_adj(self):
        with Scope("ada_vector", prefix=self.path, reuse=self.reuse) as scope:
            self.v1 = tf.get_variable(
                'v1',
                shape=[2 * self.E, self.v_dim],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            self.v2 = tf.get_variable(
                'v2',
                shape=[self.v_dim, 2 * self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            A = tf.nn.softmax(tf.nn.relu(tf.matmul(self.v1, self.v2)),
                              axis=1,
                              name="ada_adj")
        return A

    def e1_rel_gcn(self, x, A, training):
        self.gcn = GCN3(self.order, self.C, self.wd, prefix=self.path)
        x = self.gcn(x, A, training, self.reuse)
        return x

    def bn_hid(self, x, training):
        with Scope("bn_hid", prefix=self.path, reuse=self.reuse):
            self.bn_hid = tf.layers.BatchNormalization(axis=-1)
            x = self.bn_hid(x, training=training)
        return x

    def fc(self, x):
        # 2*E*C -> E
        return tf.layers.dense(
            x,
            self.E,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="fc")

    def bn_last(self, x, training):
        with Scope("bn_last", prefix=self.path, reuse=self.reuse):
            self.bn_last = tf.layers.BatchNormalization(axis=-1)
            x = self.bn_last(x, training=training)
        return x

    def __call__(self,
                 e1,
                 rel,
                 training=False,
                 pretrained_embeddings=None,
                 triple=None,
                 neg_triple=None):
        with Scope("AdaE3", prefix=self.prefix, reuse=self.reuse) as scope:

            e1_emb, rel_emb = self.embedding_lookup(e1, rel,
                                                    pretrained_embeddings)

            if triple is not None:
                dis_pos, dis_neg = self.transe(triple, neg_triple)
            else:
                dis_pos = 0
                dis_neg = 1.

            A = self.make_ada_adj()
            x = tf.concat([e1_emb, rel_emb], 1)  #[None, 1024, 1]
            x = self.e1_rel_gcn(x, A, training)
            x = tf.layers.flatten(x, name="flatten")  # b, 2E, C -> b,2E*C
            x = self.bn_hid(x, training)
            x = tf.nn.relu(x)
            x = self.fc(x)
            x = self.bn_last(x, training)
            x = tf.nn.relu(x)
            # bs E  E, N -> bs, N
            with Scope("pred", prefix=scope.prefix,
                       reuse=self.reuse) as scope_pred:
                x = tf.matmul(x, self.emb_e, transpose_b=True, name="mul_E")
                self.b = tf.get_variable(
                    'bias',
                    shape=[self.NE],
                    initializer=tf.zeros_initializer(),
                    regularizer=self.regularizer,
                )
                x = tf.add(x, self.b, name="logits")
        return x, dis_pos, dis_neg


class GCN3(object):
    def __init__(
        self,
        order=2,
        C=32,
        wd=0.0003,
        prefix="",
    ):
        super().__init__()
        self.order = order
        self.C = C
        self.prefix = prefix

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=wd)

    def __call__(
        self,
        x,
        A,
        training,
        reuse=None,
    ):
        '''
        A: n x n
        '''
        with Scope("GCN", prefix=self.prefix, reuse=reuse) as scope:
            # E = tf.shape(x_e)[1]
            xs = []

            x_identity_0 = x

            for i in range(self.order):
                with Scope(f"layer_{i+1}", prefix=scope.prefix, reuse=reuse):
                    # nxn b n e -> b n e
                    x_identity = x

                    if i < self.order - 1:
                        x = tf.layers.BatchNormalization(axis=-1)(
                            x, training=training)
                        if x != 0:
                            x = tf.nn.relu(x)

                    x = tf.matmul(A, x)
                    x = tf.layers.dense(
                        x,
                        self.C,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        kernel_regularizer=self.regularizer,
                        name=f"W{i+1}")
                    x = tf.add(x_identity, x)
                    xs.append(x)

            with Scope("readout", prefix=scope.prefix, reuse=reuse):
                h = tf.concat(xs, -1)  #[None, 1024, 32 * 2]
                h = tf.layers.BatchNormalization(axis=-1)(h, training=training)
                h = tf.relu(h)
                h = tf.layers.dense(
                    h,
                    self.C,
                    name="fc_readout",
                    activation=None,
                    use_bias=True,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=self.regularizer,
                )
                h = h + tf.layers.dense(
                    x_identity_0,
                    self.C,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    kernel_regularizer=self.regularizer)
        return h


class ConvE(object):
    def __init__(
        self,
        NE,
        NR,
        E,
        H,
        C=32,
        inp_dp=0.2,
        hid_dp=0.2,
        last_dp=0.3,
        wd=0.0003,
        prefix="",
        reuse=None,
    ):
        super().__init__()
        self.NE = NE
        self.NR = NR
        self.E = E
        self.EH = H
        self.EW = E // H
        self.C = C
        self.inp_dp = inp_dp
        self.hid_dp = hid_dp
        self.last_dp = last_dp
        self.prefix = prefix
        self.reuse = reuse

        self.path = f"{prefix}ConvE/"

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=wd)

    def embedding_lookup(self, e1, rel, pretrained_embeddings=None):
        # [NE, E] - lookup -> [bs, E] -> [bs, E, 1]
        with Scope("embedding_lookup", prefix=self.path, reuse=self.reuse):
            self.emb_e = tf.get_variable(
                'emb_e',
                shape=[self.NE, self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            self.emb_rel = tf.get_variable(
                'emb_r',
                shape=[self.NR, self.E],
                initializer=tf.glorot_normal_initializer(),
                regularizer=self.regularizer,
            )

            if pretrained_embeddings is not None:
                e_emb_extra = tf.nn.embedding_lookup(
                    pretrained_embeddings, e1, name="e1_pretrained_embedding")
                e1_emb = tf.reshape(
                    tf.add(
                        e_emb_extra,
                        tf.nn.embedding_lookup(self.emb_e,
                                               e1,
                                               name="e1_embedding")),
                    [-1, self.EH, self.EW, 1])
            else:
                e1_emb = tf.reshape(
                    tf.nn.embedding_lookup(self.emb_e, e1,
                                           name="e1_embedding"),
                    [-1, self.EH, self.EW, 1])

            rel_emb = tf.reshape(
                tf.nn.embedding_lookup(self.emb_rel, rel,
                                       name="rel_embedding"),
                [-1, self.EH, self.EW, 1])
        return e1_emb, rel_emb

    def conv(self, x):
        conv = tf.layers.Conv2D(self.C,
                                kernel_size=(3, 3),
                                kernel_regularizer=self.regularizer)
        return conv(x)

    def transe(self, triple, neg_triple):
        with Scope("transe", prefix=self.path, reuse=self.reuse):
            pos_h_e = triple[0]
            pos_r_e = triple[1]
            pos_t_e = triple[2]
            neg_h_e = neg_triple[0]
            neg_r_e = neg_triple[1]
            neg_t_e = neg_triple[2]

            pos_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_h_e,
                                             name="pos_h_e")
            pos_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             pos_r_e,
                                             name="pos_rel_e")
            pos_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             pos_t_e,
                                             name="pos_t_e")
            neg_h_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_h_e,
                                             name="neg_h_e")
            neg_r_e = tf.nn.embedding_lookup(self.emb_rel,
                                             neg_r_e,
                                             name="neg_rel_e")
            neg_t_e = tf.nn.embedding_lookup(self.emb_e,
                                             neg_t_e,
                                             name="neg_t_e")
            # l2 normalization
            pos_h_e = tf.nn.l2_normalize(pos_h_e, axis=1)
            pos_r_e = tf.nn.l2_normalize(pos_r_e, axis=1)
            pos_t_e = tf.nn.l2_normalize(pos_t_e, axis=1)
            neg_h_e = tf.nn.l2_normalize(neg_h_e, axis=1)
            neg_r_e = tf.nn.l2_normalize(neg_r_e, axis=1)
            neg_t_e = tf.nn.l2_normalize(neg_t_e, axis=1)

            dis_pos = tf.norm(pos_h_e + pos_r_e - pos_t_e, 2, keep_dims=True)
            dis_neg = tf.norm(neg_h_e + neg_r_e - neg_t_e, 2, keep_dims=True)
            return dis_pos, dis_neg

    def ic_emb(self, x, training):
        with Scope("ic_emb", prefix=self.path, reuse=self.reuse):
            self.bn = tf.layers.BatchNormalization(axis=-1, name="bn_inp")
            # self.dp = tf.layers.Dropout(rate=self.inp_dp, name="dp_inp")
            # x = self.dp(self.bn(x, training=training), training=training)
            x = self.bn(x, training=training)
        return x

    def ic_hid(self, x, training):
        with Scope("ic_hid", prefix=self.path, reuse=self.reuse):
            self.dp_hid = tf.layers.Dropout(rate=self.hid_dp)
            self.bn_hid = tf.layers.BatchNormalization(axis=-1)
            x = self.dp_hid(self.bn_hid(x, training=training),
                            training=training)
        return x

    def fc(self, x, training, name="fc"):
        with Scope(name, prefix=self.path, reuse=self.reuse):
            # 2*E*C -> E
            x = tf.layers.dense(
                x,
                self.E,
                activation=None,
                use_bias=None,
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=self.regularizer,
                name=name)
            # x = tf.layers.Dropout(rate=self.last_dp)(x, training=training)
            x = tf.layers.BatchNormalization(axis=-1,
                                             name="bn2")(x, training=training)
            x = tf.layers.Dropout(rate=self.last_dp)(x, training=training)
            x = tf.nn.relu(x)
        return x

    def ic_last(self, x, training):
        with Scope("ic_last", prefix=self.path, reuse=self.reuse):
            self.bn_last = tf.layers.BatchNormalization(axis=-1)
            self.dp_last = tf.layers.Dropout(rate=self.last_dp)
            x = self.dp_last(self.bn_last(x, training=training),
                             training=training)
        return x

    def pred(self, x, name="pred"):
        with Scope(name, prefix=self.prefix, reuse=self.reuse) as scope_pred:
            x = tf.matmul(x, self.emb_e, transpose_b=True, name="mul_E")
            b = tf.get_variable('bias',
                                shape=[self.NE],
                                initializer=tf.zeros_initializer())
            x = tf.add(x, b, name="logits")
        return x

    def __call__(self,
                 e1,
                 rel,
                 training=False,
                 pretrained_embeddings=None,
                 triple=None,
                 neg_triple=None):
        with Scope("ConvE", prefix=self.prefix, reuse=self.reuse) as scope:
            # [B, E] -> [B, EH, EW, 1]
            e1_emb, rel_emb = self.embedding_lookup(e1, rel,
                                                    pretrained_embeddings)

            if triple is not None:
                dis_pos, dis_neg = self.transe(triple, neg_triple)
            else:
                dis_pos = 0.
                dis_neg = 1.

            # [B, EH, EW, 1] -> [B, 2*EH, EW, 1]
            x = tf.concat([e1_emb, rel_emb], axis=1)
            x = self.ic_emb(x, training)
            x = self.conv(x)  # [B, EH', EW', C]
            x = tf.layers.BatchNormalization(axis=-1,
                                             name="bn1")(x, training=training)
            x = tf.nn.relu(x)
            # x = tf.layers.Dropout(rate=self.hid_dp)(x, training=training)
            x_hid = tf.layers.flatten(x, name="flatten")
            # [B, E'] -> [B, E]
            # x_hid = tf.layers.BatchNormalization(axis=-1, name="bn_hid")(
            #     x_hid, training=training)
            # x0 = self.fc(x_hid, training=training, name="fc_label")
            x1 = self.fc(x_hid, training=training)
            x1 = tf.layers.BatchNormalization(axis=-1, name="bn_last")(
                x1, training=training)
            x1 = tf.nn.relu(x1)
            # logit_label = self.pred(x0, name="pred_label")
            logit = self.pred(x1)
        # return (logit_label, logit), dis_pos, dis_neg
        return logit, dis_pos, dis_neg


def test():
    set_gpu(3)
    # test_gcn()
    test_ada()


def test_gcn():
    B = 11
    C = 8
    gcn = GCN(2, C)
    # x_e = tf.placeholder(tf.int64, name="x_e", shape=[None, C, 1])
    # x_rel = tf.placeholder(tf.int64, name="x_rel", shape=[None, C, 1])

    with tf.variable_scope("input"):
        x_e = tf.random_normal([B, C, 1], name="x_e")
        x_rel = tf.random_normal([B, C, 1], name="x_rel")
        A = tf.random_normal([2 * C, 2 * C], name="A")

    y = gcn(x_e, x_rel, A)

    print(y)

    write_graph("gcn")


def test_ada():
    B = 11
    NE = 22
    NR = 3
    C = 8

    ada = AdaE(NE, NR, C)

    with tf.name_scope("input"):
        ph_rel = tf.placeholder(tf.int64, shape=[None], name="rel")
        ph_e1 = tf.placeholder(tf.int64, shape=[None], name="e1")
        ph_training = tf.placeholder(tf.bool, name="training")

    y = ada(ph_e1, ph_rel, ph_training)
    write_graph("AdaE")
    print(y)

    with tf.Session() as sess:
        e1 = np.random.randint(0, NE, (B, ))
        rel = np.random.randint(0, NR, (B, ))
        sess.run(tf.global_variables_initializer())
        y = sess.run(y, feed_dict={ph_e1: e1, ph_rel: rel, ph_training: 1})
        print(y)
        print(y.shape)


class Export(object):
    def __init__(self,
                 NE,
                 NR,
                 E,
                 C=32,
                 v_dim=10,
                 output_dir="export",
                 H=32,
                 wd=0.0003,
                 model_name="AdaE",
                 use_transe2=False,
                 use_other_loss=False,
                 use_masked_label=True,
                 q=0.7,
                 alpha=0.1,
                 beta=1,
                 A=1e-4):
        super().__init__()
        self.NE = NE
        self.NR = NR
        self.E = E
        self.C = C
        self.H = H
        self.v_dim = v_dim
        self.wd = wd

        self.output_dir = output_dir + f"/{model_name}"
        self.model_name = model_name
        self.use_transe2 = use_transe2
        self.use_other_loss = use_other_loss
        self.use_masked_label = use_masked_label
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.A = A
        print(f"Export::use_transe2: {self.use_transe2}")

    def build(self):
        print("Export::build...")
        self._build_ph()
        self._build_predict()
        # must be the last
        self._build_init()

    def _build_init(self):
        print("build_init...")
        print("global variables:", tf.global_variables())
        self.init = tf.variables_initializer(tf.global_variables(),
                                             name='init')

    def _build_ph(self):
        with tf.name_scope("input"):
            self.ph_rel = tf.placeholder(tf.int64, shape=[None], name="rel")
            self.ph_e1 = tf.placeholder(tf.int64, shape=[None], name="e1")

    def _build_predict(self):
        logits, dis_pos, dis_neg = self._build_forward(self.ph_e1, self.ph_rel,
                                                       False, False)

        prediction = tf.nn.sigmoid(logits, name="prediction")
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_forward(self,
                       e1,
                       rel,
                       training,
                       reuse=None,
                       triple=None,
                       neg_triple=None):
        if training:
            with Scope("forward", reuse=reuse) as scope:
                if self.model_name == "AdaE":
                    print("model is AdaE")
                    model = AdaE(self.NE,
                                 self.NR,
                                 self.E,
                                 self.C,
                                 v_dim=self.v_dim,
                                 prefix=scope.prefix)
                elif self.model_name == "AdaE2":
                    print("model is AdaE2")
                    model = AdaE2(self.NE,
                                  self.NR,
                                  self.E,
                                  self.C,
                                  v_dim=self.v_dim,
                                  wd=self.wd,
                                  prefix=scope.prefix)
                elif self.model_name == "AdaE3":
                    print("model is AdaE3")
                    model = AdaE3(self.NE,
                                  self.NR,
                                  self.E,
                                  self.C,
                                  v_dim=self.v_dim,
                                  wd=self.wd,
                                  prefix=scope.prefix)
                else:
                    print("model is ConvE")
                    model = ConvE(self.NE,
                                  self.NR,
                                  self.E,
                                  self.H,
                                  self.C,
                                  wd=self.wd,
                                  prefix=scope.prefix)

                logits, dis_pos, dis_neg = model(e1, rel, training, None,
                                                 triple, neg_triple)
            return logits, dis_pos, dis_neg
        else:
            with Scope("forward", reuse=reuse) as scope:
                if self.model_name == "AdaE":
                    model = AdaE(self.NE,
                                 self.NR,
                                 self.E,
                                 self.C,
                                 v_dim=self.v_dim,
                                 reuse=True,
                                 prefix=scope.prefix)
                elif self.model_name == "AdaE2":
                    print("model is AdaE2")
                    model = AdaE2(self.NE,
                                  self.NR,
                                  self.E,
                                  self.C,
                                  v_dim=self.v_dim,
                                  wd=self.wd,
                                  prefix=scope.prefix)
                elif self.model_name == "AdaE3":
                    print("model is AdaE3")
                    model = AdaE3(self.NE,
                                  self.NR,
                                  self.E,
                                  self.C,
                                  v_dim=self.v_dim,
                                  wd=self.wd,
                                  prefix=scope.prefix)
                else:
                    model = ConvE(self.NE,
                                  self.NR,
                                  self.E,
                                  self.H,
                                  self.C,
                                  reuse=True,
                                  wd=self.wd,
                                  prefix=scope.prefix)
                logits, dis_pos, dis_neg = model(e1, rel, training, None,
                                                 triple, neg_triple)
            return logits, dis_pos, dis_neg

    def export(self):
        self.build()

        definition = tf.Session().graph_def
        tf.io.write_graph(definition,
                          "export/AdaE2_SaveModel",
                          'model_txt.pb',
                          as_text=True)
        # tf.io.write_graph(definition,
        #                   self.output_dir,
        #                   'model.pb',
        #                   as_text=False)
        # write_graph(self.model_name)
        write_graph("AdaE2_SaveModel")


def build_graph(E: int = 512,
                C: int = 32,
                v_dim: int = 16,
                H: int = 32,
                wd: float = 0.0003,
                model_name: str = "AdaE",
                use_transe2: bool = False,
                use_masked_label=True,
                q=0.7,
                alpha=0.1,
                beta=1.0,
                A=1e-4) -> None:
    # print(f"build_graph:: model: {model_name} use_transe: {use_transe2}")
    print(f"build_graph:: model: {model_name}")
    if model_name == "AdaE" or model_name == "AdaE2" or model_name == "AdaE3":
        export = Export(
            NE,
            NR,
            E,
            C,
            v_dim,
            wd=wd,
            model_name=model_name,
            use_transe2=use_transe2,
            use_masked_label=use_masked_label,
            q=q,
            alpha=alpha,
            beta=-beta,
            A=A,
        )
    else:
        export = Export(
            NE,
            NR,
            E,
            C,
            H=H,
            wd=wd,
            model_name=model_name,
            use_transe2=use_transe2,
            use_masked_label=use_masked_label,
            q=q,
            alpha=alpha,
            beta=-beta,
            A=A,
        )
    # export.build()
    export.export()


def export(model_name="AdaE", use_transe2=False):
    set_gpu(1)
    NE = 28754
    NR = 10
    E = 512
    C = 32
    v_dim = 16
    H = 32

    if model_name == "AdaE":
        export = Export(NE,
                        NR,
                        E,
                        C,
                        v_dim,
                        model_name=model_name,
                        use_transe2=use_transe2)
    else:
        export = Export(NE,
                        NR,
                        E,
                        C,
                        H=H,
                        model_name=model_name,
                        use_transe2=use_transe2)

    export.export()


if __name__ == "__main__":
    # test()
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        use_transe2 = False
        if len(sys.argv) > 2:
            use_transe2 = True
        export(model_name, use_transe2)
    else:
        export()
