import numpy as np
import tensorflow as tf

from const import NE, NR, TEST_SIZE, TRAIN_SIZE, VAL_SIZE
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
                initializer=tf.glorot_uniform_initializer())

            self.emb_rel = tf.get_variable(
                'emb_r',
                shape=[self.NR, self.E],
                initializer=tf.glorot_uniform_initializer())

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

            e1_emb, rel_emb = self.ic_emb(e1_emb, rel_emb, training)
            A = self.make_ada_adj()
            x = self.e1_rel_gcn(e1_emb, rel_emb, A)

            x = tf.nn.relu(x)
            x = self.ic_hid(x, training)
            x = tf.layers.flatten(x, name="flatten")  # b, 2E, C -> b,2E*C

            x = tf.nn.relu(x)
            x = self.fc(x)

            x = self.ic_last(x, training)
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


class AdaExport(object):
    def __init__(self, NE, NR, E, C=32, v_dim=10, output_dir="export"):
        super().__init__()
        self.NE = NE
        self.NR = NR
        self.E = E
        self.C = C
        self.v_dim = v_dim

        self.output_dir = output_dir

    def build(self):
        self._build_ph()
        self._build_train()
        self._build_summary()
        self._build_summary_gards()
        # must be the last
        self._build_print_data()
        self._build_init()

    def _build_init(self):
        self.init = tf.variables_initializer(tf.global_variables(),
                                             name='init')

    def _build_ph(self):
        with tf.name_scope("custom"):
            self.batch_size_trn = tf.placeholder(tf.int64, (),
                                                 "batch_size_trn")
            self.batch_size_dev = tf.placeholder(tf.int64, (),
                                                 "batch_size_dev")
            self.repeat = tf.placeholder(tf.int64, (), "repeat")
            self.lr = tf.placeholder_with_default(0.001, (), "lr")
            # dataset
            self.train_size = tf.placeholder_with_default(
                tf.constant(TRAIN_SIZE, tf.int64), (), "train_size")

            self.val_size = tf.placeholder_with_default(
                tf.constant(VAL_SIZE, tf.int64), (), "val_size")
            self.test_size = tf.placeholder_with_default(
                tf.constant(TEST_SIZE, tf.int64), (), "test_size")

            # pre-trained word embedding
            # self.pretrained_embeddings = tf.placeholder(tf.float32, (NE, 512), "pretrained_embeddings")
            self.pretrained_embeddings = tf.placeholder_with_default(
                tf.zeros((NE, 512)), (NE, 512), "pretrained_embeddings")
            self.use_transe = tf.placeholder_with_default(
                False, (), "use_transe")

    def _build_print_data(self):
        self._build_print_data_trn()
        self._build_print_data_val()
        self._build_print_data_test()

    def _build_print_data_trn(self):
        with tf.name_scope("print_data_trn"):
            triple = self.batch_data_trn["triple"].values.values
            neg_triple = self.batch_data_trn["neg_triple"].values.values

            inp_trn = self.batch_data_trn["input"]
            inp_trn.set_shape((None, 2))
            label_trn = self.batch_data_trn["label"]

            print_op = tf.print(inp_trn,
                                "\n",
                                "\n",
                                "\ntriple:\n",
                                triple,
                                "\n",
                                "\nneg_triple:\n",
                                neg_triple,
                                "\n",
                                "\n",
                                type(triple),
                                type(self.batch_data_trn["triple"]),
                                tf.shape(triple),
                                tf.shape(neg_triple),
                                tf.reduce_sum(label_trn),
                                name="print")

    def _build_print_data_val(self):
        with tf.name_scope("print_data_val"):
            triple = self.batch_data_val["triple"].values.values
            neg_triple = self.batch_data_val["neg_triple"].values.values

            inp_val = self.batch_data_val["input"]
            inp_val.set_shape((None, 2))
            label_val = self.batch_data_val["label"]

            print_op = tf.print(inp_val,
                                "\n",
                                "\n",
                                "\ntriple:\n",
                                triple,
                                "\n",
                                "\nneg_triple:\n",
                                neg_triple,
                                "\n",
                                "\n",
                                type(triple),
                                type(self.batch_data_val["triple"]),
                                tf.shape(triple),
                                tf.shape(neg_triple),
                                tf.reduce_sum(label_val),
                                name="print")

    def _build_print_data_test(self):
        with tf.name_scope("print_data_test"):
            triple = self.batch_data_test["triple"].values.values
            neg_triple = self.batch_data_test["neg_triple"].values.values

            inp_test = self.batch_data_test["input"]
            inp_test.set_shape((None, 2))
            label_test = self.batch_data_test["label"]

            print_op = tf.print(inp_test,
                                "\n",
                                "\n",
                                "\ntriple:\n",
                                triple,
                                "\n",
                                "\nneg_triple:\n",
                                neg_triple,
                                "\n",
                                "\n",
                                type(triple),
                                type(self.batch_data_test["triple"]),
                                tf.shape(triple),
                                tf.shape(neg_triple),
                                tf.reduce_sum(label_test),
                                name="print")

    def _build_train(self):
        with tf.name_scope("data_trn"):
            _iterator_trn, batch_data_trn = self._build_train_dataset(
                self.repeat, batch_size=self.batch_size_trn)
            self.batch_data_trn = batch_data_trn

            triple_trn = batch_data_trn["triple"].values.values
            neg_triple_trn = batch_data_trn["neg_triple"].values.values

            inp_trn = batch_data_trn["input"]
            inp_trn.set_shape((None, 2))
            label_trn = batch_data_trn["label"]
            e1_trn, rel_trn = tf.unstack(inp_trn, axis=1)

        logits, dis_pos, dis_neg = self._build_forward(e1_trn, rel_trn, True,
                                                       None, triple_trn,
                                                       neg_triple_trn)
        self._build_loss(label_trn, logits, dis_pos, dis_neg)
        self._build_optimizer()

        # eval
        with tf.name_scope("data_val"):
            _iterator_val, batch_data_val = self._build_dev_dataset(
                self.val_size, dev="val", batch_size=self.batch_size_dev)
            self.batch_data_val = batch_data_val

            triple_val = batch_data_val["triple"].values.values
            neg_triple_val = batch_data_val["neg_triple"].values.values

            inp_val = batch_data_val["input"]
            inp_val.set_shape((None, 2))
            label_val = batch_data_val["label"]
            aux_label_val = batch_data_val["aux_label"]
            e1_val, rel_val = tf.unstack(inp_val, axis=1)

        logits_val, _, _ = self._build_forward(e1_val, rel_val, False, True,
                                               triple_val, neg_triple_val)

        _, _, _, self.rank_val, self.hits1_val, self.hits3_val, self.hits10_val, _ = self._build_eval(
            logits_val,
            label_val,
            aux_label_val,
            self.batch_size_dev,
            name_suffix="val")

        with tf.name_scope("data_test"):
            _iterator_test, batch_data_test = self._build_dev_dataset(
                self.test_size, dev="test", batch_size=self.batch_size_dev)
            self.batch_data_test = batch_data_test

            triple_test = batch_data_test["triple"].values.values
            neg_triple_test = batch_data_test["neg_triple"].values.values

            inp_test = batch_data_test["input"]
            inp_test.set_shape((None, 2))
            label_test = batch_data_test["label"]
            aux_label_test = batch_data_test["aux_label"]
            e1_test, rel_test = tf.unstack(inp_test, axis=1)

        logits_test, _, _ = self._build_forward(e1_test, rel_test, False, True,
                                                triple_test, neg_triple_test)

        _, _, _, self.rank_test, self.hits1_test, self.hits3_test, self.hits10_test, _ = self._build_eval(
            logits_test,
            label_test,
            aux_label_test,
            self.batch_size_dev,
            name_suffix="test")

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_train_dataset(self, repeat_num, batch_size=4):
        with tf.name_scope('dataset_trn'):
            dataset_trn = tf.data.TFRecordDataset("symptom_trn.tfrecord",
                                                  num_parallel_reads=16)

            # Create a description of the features.
            feature_description_trn = {
                'input':
                tf.io.FixedLenSequenceFeature([],
                                              tf.int64,
                                              default_value=0,
                                              allow_missing=True),
                'label':
                tf.io.FixedLenSequenceFeature([],
                                              tf.int64,
                                              default_value=0,
                                              allow_missing=True),
                'num_ent':
                tf.io.FixedLenSequenceFeature([],
                                              tf.int64,
                                              default_value=0,
                                              allow_missing=True),
            }

            def _parse_function_trn(example_proto):
                return tf.io.parse_single_example(example_proto,
                                                  feature_description_trn)

            def _transform(d):
                with tf.name_scope("transform"):
                    zeros = tf.zeros(self.NE, name="zeros")
                    label = tf.convert_to_tensor(d["label"], dtype=tf.int64)
                    d["num_label"] = tf.shape(label)
                    label = scatter_update_tensor(
                        zeros, tf.cast(tf.expand_dims(label, axis=1),
                                       tf.int32),
                        tf.ones_like(label, dtype=tf.float32))
                    d["label"] = label
                    return d

            def _transform1(d):
                with tf.name_scope("transform"):
                    zeros = tf.zeros(self.NE, name="zeros")
                    label = tf.convert_to_tensor(d["label"], dtype=tf.int64)
                    d["num_label"] = tf.shape(label)
                    label = scatter_update_tensor(
                        zeros, tf.cast(tf.expand_dims(label, axis=1),
                                       tf.int32),
                        tf.ones_like(label, dtype=tf.float32))
                    d["label"] = label
                    return d

            def _transform2(d):
                with tf.name_scope("transform"):
                    label = tf.convert_to_tensor(d["label"], dtype=tf.int64)
                    zeros = tf.zeros(self.NE, name="zeros")
                    label = scatter_update_tensor(
                        zeros, tf.cast(tf.expand_dims(label, axis=1),
                                       tf.int32),
                        tf.ones_like(label, dtype=tf.float32))
                    d["label"] = label

                    ents = tf.range(0, self.NE, dtype=tf.int64, name="ents")
                    pos_label_num = tf.cast(tf.reduce_sum(label), tf.int64)
                    pos_label = tf.boolean_mask(ents, tf.cast(label, tf.bool))
                    neg_label = tf.boolean_mask(ents,
                                                tf.cast(1 - label, tf.bool))
                    neg_label_num = tf.subtract(tf.cast(self.NE, tf.int64),
                                                pos_label_num)
                    rand_idx = tf.cast(
                        tf.random.uniform((pos_label_num, ), 0,
                                          tf.cast(neg_label_num, tf.float32)),
                        tf.int64)
                    neg_label = tf.gather(neg_label, rand_idx)

                    hrs = tf.broadcast_to(d["input"], (pos_label_num, 2))
                    pos_labels = tf.expand_dims(pos_label, axis=1)
                    triple = tf.concat((hrs, pos_labels),
                                       axis=1,
                                       name="triple")

                    neg_labels = tf.expand_dims(neg_label, axis=1)
                    neg_triple = tf.concat((hrs, neg_labels),
                                           axis=1,
                                           name="neg_triple")

                    d["triple"] = tf.RaggedTensor.from_tensor(
                        tf.expand_dims(triple, axis=0), name="triple_trn")
                    d["neg_triple"] = tf.RaggedTensor.from_tensor(
                        tf.expand_dims(neg_triple, axis=0),
                        name="neg_triple_trn")

                    # d["num_label"] = tf.shape(label)
                    # d["pos_label_num"] = pos_label_num

                    # foo = tf.random.uniform((5, ), 0, 10)
                    # d["foo"] = foo
                    # d["bar"] = tf.add(foo, 20.)

                    return d

            # dataset_trn = dataset_trn.map(_parse_function_trn,
            #                               num_parallel_calls=16).map(
            #                                   _transform,
            #                                   num_parallel_calls=16)
            dataset_trn = dataset_trn.map(_parse_function_trn,
                                          num_parallel_calls=16).map(
                                              _transform2,
                                              num_parallel_calls=16)

            iterator_trn = dataset_trn.take(self.train_size).shuffle(
                tf.cast((tf.cast(self.train_size, tf.float32) * 1.2),
                        tf.int64)).repeat(repeat_num).batch(
                            batch_size, drop_remainder=True).prefetch(
                                batch_size).make_initializable_iterator()

            # iterator_trn = dataset_trn.take(
            #     self.train_size).repeat(repeat_num).batch(
            #         batch_size, drop_remainder=True).prefetch(
            #             batch_size).make_initializable_iterator()
            # iterator = dataset.shuffle(30000).repeat(10).batch(4).prefetch(4).make_one_shot_iterator()

            batch_data_trn = iterator_trn.get_next("get_next")
        return iterator_trn, batch_data_trn

    def _build_dev_dataset(self, dev_size, dev="test", batch_size=4):
        with tf.name_scope(f"dataset_{dev}"):
            dataset_dev = tf.data.TFRecordDataset(f"symptom_{dev}.tfrecord",
                                                  num_parallel_reads=16)
            feature_description_dev = {
                'input':
                tf.io.FixedLenSequenceFeature([],
                                              tf.int64,
                                              default_value=0,
                                              allow_missing=True),
                # 'num_ent':
                # tf.io.FixedLenSequenceFeature([],
                #                               tf.int64,
                #                               default_value=0,
                #                               allow_missing=True),
                'num_ent':
                tf.io.FixedLenFeature(
                    [1],
                    tf.int64,
                    default_value=0,
                ),
                'label':
                tf.io.FixedLenSequenceFeature([],
                                              tf.int64,
                                              default_value=0,
                                              allow_missing=True),
                'aux_label':
                tf.io.FixedLenSequenceFeature([],
                                              tf.int64,
                                              default_value=0,
                                              allow_missing=True),
            }

            def _parse_function_dev(example_proto):
                return tf.io.parse_single_example(example_proto,
                                                  feature_description_dev)

            def _transform(d):
                with tf.name_scope("transform"):
                    zeros = tf.zeros(self.NE, name="zeros")

                    aux_label = tf.convert_to_tensor(d["aux_label"],
                                                     dtype=tf.int64)
                    # [m,] => [m, 1]
                    aux_label = scatter_update_tensor(
                        zeros,
                        tf.cast(tf.expand_dims(aux_label, axis=1), tf.int32),
                        tf.ones_like(aux_label, dtype=tf.float32))
                    # soft_label = tf.divide(1., 1. - eps) * label + tf.divide(1.0, N)
                    d["aux_label"] = aux_label
                    d["label"] = tf.RaggedTensor.from_tensor(
                        tf.expand_dims(d["label"], axis=0))
                    return d

            def _transform2(d):
                with tf.name_scope("transform"):
                    zeros = tf.zeros(self.NE, name="zeros")
                    # aux_label [3, 4, 6] and label: [3]
                    # boolean mask from aux_label and generate label's positive number sample
                    label = tf.convert_to_tensor(d["label"], dtype=tf.int64)

                    aux_label = tf.convert_to_tensor(d["aux_label"],
                                                     dtype=tf.int64)
                    # [m,] => [m, 1]
                    aux_label = scatter_update_tensor(
                        zeros,
                        tf.cast(tf.expand_dims(aux_label, axis=1), tf.int32),
                        tf.ones_like(aux_label, dtype=tf.float32))
                    # soft_label = tf.divide(1., 1. - eps) * label + tf.divide(1.0, N)

                    ents = tf.range(0, self.NE, dtype=tf.int64, name="ents")
                    pos_label = label
                    pos_label_num = tf.cast(tf.shape(label)[0], tf.int64)
                    neg_label = tf.boolean_mask(
                        ents, tf.cast(1 - aux_label, tf.bool))
                    neg_label_num = tf.subtract(
                        tf.cast(self.NE, tf.int64),
                        tf.cast(tf.reduce_sum(aux_label), tf.int64))
                    rand_idx = tf.cast(
                        tf.random.uniform((pos_label_num, ), 0,
                                          tf.cast(neg_label_num, tf.float32)),
                        tf.int64)
                    neg_label = tf.gather(neg_label, rand_idx)

                    hrs = tf.broadcast_to(d["input"], (pos_label_num, 2))
                    pos_labels = tf.expand_dims(pos_label, axis=1)
                    triple = tf.concat((hrs, pos_labels),
                                       axis=1,
                                       name="triple")
                    d["triple"] = tf.RaggedTensor.from_tensor(
                        tf.expand_dims(triple, axis=0))

                    neg_labels = tf.expand_dims(neg_label, axis=1)
                    neg_triple = tf.concat((hrs, neg_labels),
                                           axis=1,
                                           name="neg_triple")
                    d["neg_triple"] = tf.RaggedTensor.from_tensor(
                        tf.expand_dims(neg_triple, axis=0))

                    d["aux_label"] = aux_label
                    d["label"] = tf.RaggedTensor.from_tensor(
                        tf.expand_dims(d["label"], axis=0))
                    return d

            # dataset_dev = dataset_dev.map(_parse_function_dev,
            #                               num_parallel_calls=16).map(
            #                                   _transform,
            #                                   num_parallel_calls=16)
            dataset_dev = dataset_dev.map(_parse_function_dev,
                                          num_parallel_calls=16).map(
                                              _transform2,
                                              num_parallel_calls=16)
            iterator_dev = dataset_dev.take(dev_size).batch(
                batch_size, drop_remainder=True).prefetch(
                    batch_size).make_initializable_iterator()
            batch_data_dev = iterator_dev.get_next("get_next")
        return iterator_dev, batch_data_dev

    def _build_forward(self,
                       e1,
                       rel,
                       training,
                       reuse=None,
                       triple=None,
                       neg_triple=None):
        if training:
            with Scope("forward", reuse=reuse) as scope:
                self.ada = AdaE(self.NE,
                                self.NR,
                                self.E,
                                self.C,
                                v_dim=self.v_dim,
                                prefix=scope.prefix)
                logits, dis_pos, dis_neg = self.ada(e1, rel, training,
                                                    self.pretrained_embeddings,
                                                    triple, neg_triple)
            return logits, dis_pos, dis_neg
        else:
            with Scope("forward", reuse=reuse) as scope:
                self.ada = AdaE(self.NE,
                                self.NR,
                                self.E,
                                self.C,
                                v_dim=self.v_dim,
                                reuse=True,
                                prefix=scope.prefix)
                logits, dis_pos, dis_neg = self.ada(e1, rel, training,
                                                    self.pretrained_embeddings,
                                                    triple, neg_triple)
            return logits, dis_pos, dis_neg

    def _build_loss(self, labels, logits, dis_pos, dis_neg, margin=1.0):
        with tf.name_scope('loss'):
            self.loss_adae = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(
                multi_class_labels=labels, logits=logits, label_smoothing=0.1),
                                           name="loss_adae")
            self.loss_margin = tf.reduce_mean(tf.maximum(
                dis_pos - dis_neg + margin, 0),
                                              name="loss_margin")
            self.loss = tf.add(self.loss_adae, self.loss_margin, name="loss")

    def _build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0,
                                           dtype=tf.int64,
                                           trainable=False,
                                           name='global_step')
            # self.optimize = tf.train.AdamOptimizer(self.lr).minimize(
            #     self.loss, self.global_step, name="optimize")
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            with tf.control_dependencies(
                [tf.add(tf.constant(1, dtype=tf.int64), self.global_step)]):
                self.optimize = self.optimizer.apply_gradients(
                    self.grads_and_vars, name="optimize")

            self.optimizer_sgd = tf.train.GradientDescentOptimizer(self.lr)
            self.grads_and_vars = self.optimizer_sgd.compute_gradients(
                self.loss)
            with tf.control_dependencies(
                [tf.add(tf.constant(1, dtype=tf.int64), self.global_step)]):
                self.optimize_sgd = self.optimizer_sgd.apply_gradients(
                    self.grads_and_vars, name="optimize_sgd")

    def _build_summary(self):
        with tf.name_scope('summaries'):
            loss_s = tf.summary.scalar('loss/loss', self.loss)
            # hloss_s = tf.summary.histogram('histogram loss', self.loss)

            loss_adae_s = tf.summary.scalar('loss/adae', self.loss_adae)
            loss_margin_s = tf.summary.scalar('loss/margin', self.loss_margin)
            self.summary_op = tf.summary.merge(
                [loss_s, loss_adae_s, loss_margin_s], name="summary_op")

            self.ph_rank_val = tf.placeholder(tf.float32, (), "rank_val")
            self.ph_hit1_val = tf.placeholder(tf.float32, (), "hit1_val")
            self.ph_hit3_val = tf.placeholder(tf.float32, (), "hit3_val")
            self.ph_hit10_val = tf.placeholder(tf.float32, (), "hit10_val")
            rank_s = tf.summary.scalar('rank', self.ph_rank_val)
            hit1_s = tf.summary.scalar('hit1', self.ph_hit1_val)
            hit3_s = tf.summary.scalar('hit3', self.ph_hit3_val)
            hit10_s = tf.summary.scalar('hit10', self.ph_hit10_val)
            self.summary_val_op = tf.summary.merge(
                [rank_s, hit1_s, hit3_s, hit10_s], name="summary_val_op")

    def _build_summary_gards(self):
        with tf.name_scope('gradients'):
            summaries = []
            for grad, var in self.grads_and_vars:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name, grad))
            self.summary_grads = tf.summary.merge(summaries,
                                                  name="summary_grads_op")

    #TODO: export sorted_idx or pred_ents
    def _build_eval(self,
                    logits,
                    labels,
                    auxes,
                    batch_size=3,
                    name_suffix="val"):
        with tf.device('/cpu:0'):
            with tf.variable_scope(f"eval_{name_suffix}"):
                preds = tf.nn.sigmoid(logits, name="preds")

                i = tf.constant(0, dtype=tf.int64)
                arr_idx = tf.constant(0, dtype=tf.int64)
                ranks = tf.TensorArray(tf.int32,
                                       size=1,
                                       element_shape=(),
                                       dynamic_size=True,
                                       name=f"ranks_{name_suffix}")

                scores = preds * tf.subtract(1.0, tf.cast(auxes, tf.float32))

                # preds is tensor(readout)
                def cond(i, arr_idx, preds, scores, labels, ranks):
                    return tf.less(i, batch_size)

                def loop_body(i, arr_idx, preds, scores, labels, ranks):
                    # the i-th row
                    # build once
                    pred = preds[i]
                    score = scores[i]
                    label = labels[i][0]  #[0] for ragged extra axis

                    # print_op = tf.print("===\n", i, label)

                    # with tf.control_dependencies([print_op]):
                    row_len = tf.cast(tf.shape(label)[0], tf.int64)

                    def cond_inner(j, i, row_len, arr_idx, pred, score, label,
                                   ranks):
                        return tf.less(j, row_len)

                    def loop_body_inner(j, i, row_len, arr_idx, pred, score,
                                        label, ranks):
                        # build once
                        l = label[j]
                        # pick out cur position
                        p = pred[l]
                        # op3 = tf.print("p shape: ", tf.shape(p))
                        # op4 = tf.print("p : ", p, "\n", "label: ", label, "l: ", l)
                        # # race?
                        # print_op = tf.print("score1", score)

                        # with tf.control_dependencies([print_op, op3, op4]):
                        indices = tf.cast(tf.broadcast_to(l, [1, 1]), tf.int32)
                        score_l = scatter_update_tensor(score, indices, [p])

                        # print_op = tf.print("score2", score_l)

                        # with tf.control_dependencies([print_op]):
                        sorted_idx = tf.argsort(score_l,
                                                direction="DESCENDING")
                        # print_op = tf.print("sorted_idx", sorted_idx)

                        # with tf.control_dependencies([print_op]):
                        rank = tf.add(tf.argsort(sorted_idx)[l], 1)
                        # print_rank = tf.print("==== rank ===: ", rank)

                        # with tf.control_dependencies([print_rank]):
                        ranks = ranks.write(tf.cast(arr_idx, tf.int32), rank)

                        return (tf.add(j, 1), i, row_len, tf.add(arr_idx, 1),
                                pred, score, label, ranks)

                    j = tf.constant(0, dtype=tf.int64)

                    # 2)
                    j, i, row_len, arr_idx, pred, score, label, ranks_inner = tf.while_loop(
                        cond_inner,
                        loop_body_inner,
                        (j, i, row_len, arr_idx, pred, score, label, ranks),
                        parallel_iterations=16)

                    return (tf.add(i, 1), arr_idx, preds, scores, labels,
                            ranks_inner)

                # 1)
                i, arr_idx, preds, scores, labels, ranks_out = tf.while_loop(
                    cond,
                    loop_body, [i, arr_idx, preds, scores, labels, ranks],
                    parallel_iterations=16)

                # ranks -> hits 1/3/10
                ranks_tensor = ranks_out.stack()

                rank = tf.reduce_mean(tf.cast(ranks_tensor, tf.float32),
                                      name=f"rank_{name_suffix}")
                hits1 = tf.reduce_mean(tf.where(
                    tf.less(ranks_tensor, 2),
                    tf.ones_like(ranks_tensor, dtype=tf.float32),
                    tf.zeros_like(ranks_tensor, dtype=tf.float32)),
                                       name=f"hits1_{name_suffix}")
                hits3 = tf.reduce_mean(tf.where(
                    tf.less(ranks_tensor, 4),
                    tf.ones_like(ranks_tensor, dtype=tf.float32),
                    tf.zeros_like(ranks_tensor, dtype=tf.float32)),
                                       name=f"hits3_{name_suffix}")
                hits10 = tf.reduce_mean(tf.where(
                    tf.less(ranks_tensor, 11),
                    tf.ones_like(ranks_tensor, dtype=tf.float32),
                    tf.zeros_like(ranks_tensor, dtype=tf.float32)),
                                        name=f"hits10_{name_suffix}")

                eval_op = tf.group(rank,
                                   hits1,
                                   hits3,
                                   hits10,
                                   name=f"eval_op_{name_suffix}")

        return preds, scores, ranks_tensor, rank, hits1, hits3, hits10, eval_op

    def export(self):
        self.build()

        definition = tf.Session().graph_def
        tf.io.write_graph(definition,
                          self.output_dir,
                          'model_txt.pb',
                          as_text=True)
        # tf.io.write_graph(definition,
        #                   self.output_dir,
        #                   'model.pb',
        #                   as_text=False)
        write_graph("AdaExport")


def build_graph(E: int = 512, C: int = 32, v_dim: int = 16) -> None:
    ada_export = AdaExport(NE, NR, E, C, v_dim)
    ada_export.build()


def export():
    set_gpu(1)
    NE = 28754
    NR = 10
    E = 512
    C = 32
    v_dim = 16
    ada_export = AdaExport(NE, NR, E, C, v_dim)
    ada_export.export()


if __name__ == "__main__":
    # test()
    # tf.reset_default_graph()
    export()
