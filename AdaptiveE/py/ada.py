import tensorflow as tf
import numpy as np
from utils import set_gpu, write_graph
# from utils import embedding as utils_embedding

# build create static ops
# call create ops dependent on logic


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
        self.reuse = reuse

    def embedding_lookup(self, e1, rel):
        # [NE, E] - lookup -> [bs, E] -> [bs, E, 1]
        with tf.variable_scope("embedding_lookup"):
            self.emb_e = tf.get_variable(
                'emb_e',
                shape=[self.NE, self.E],
                initializer=tf.glorot_uniform_initializer())

            self.emb_rel = tf.get_variable(
                'emb_r',
                shape=[self.NR, self.E],
                initializer=tf.glorot_uniform_initializer())
            e1_emb = tf.reshape(
                tf.nn.embedding_lookup(self.emb_e, e1, name="e1_embedding"),
                [-1, self.E, 1])
            rel_emb = tf.reshape(
                tf.nn.embedding_lookup(self.emb_rel, rel,
                                       name="rel_embedding"), [-1, self.E, 1])
            return e1_emb, rel_emb

    def ic_emb(self, e1_emb, rel_emb, training):
        with tf.variable_scope("ic_emb"):
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
        with tf.variable_scope('ada_vector'):
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
        self.gcn = GCN(self.order, self.C)
        x = self.gcn(e1_emb, rel_emb, A)
        return x

    def ic_hid(self, x, training):
        with tf.variable_scope("ic_hid"):
            self.bn_hid = tf.layers.BatchNormalization(axis=-1)
            self.dp_hid = tf.layers.Dropout(rate=self.hid_dp)
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
        with tf.variable_scope("ic_last"):
            self.bn_last = tf.layers.BatchNormalization(axis=-1)
            self.dp_last = tf.layers.Dropout(rate=self.last_dp)
            x = self.dp_last(self.bn_last(x, training=training),
                             training=training)
        return x

    def __call__(self, e1, rel, training=False):
        with tf.variable_scope("AdaE", reuse=self.reuse):
            e1_emb, rel_emb = self.embedding_lookup(e1, rel)
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
            with tf.variable_scope("pred"):
                x = tf.matmul(x, self.emb_e, transpose_b=True, name="mul_E")
                self.b = tf.get_variable('bias',
                                         shape=[self.NE],
                                         initializer=tf.zeros_initializer())
                x = tf.add(x, self.b, name="logits")
            return x


class GCN(object):
    def __init__(self, order=2, C=32):
        super().__init__()
        self.order = order
        self.C = C

    def __call__(self, x_e, x_rel, A):
        '''
        A: n x n
        '''
        with tf.variable_scope("GCN"):
            # E = tf.shape(x_e)[1]
            xs = []

            with tf.variable_scope(f"layer_0"):
                x = tf.concat([x_e, x_rel], 1)

            for i in range(self.order):
                with tf.variable_scope(f"layer_{i+1}"):
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

            with tf.variable_scope("readout"):
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
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())


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
    def __init__(self, NE, NR, C, output_dir="export"):
        super().__init__()
        self.NE = NE
        self.NR = NR
        self.C = C
        self.output_dir = output_dir

    def build(self):
        self._build_input_placeholder()
        self._build_forward()
        self._build_loss()
        self._build_optimizer()
        self._build_summary()
        self._build_train()
        self._build_input_data()

    def _build_input_placeholder(self):
        with tf.name_scope("input"):
            self.e1 = tf.placeholder(tf.int64, shape=[None], name="e1")
            self.rel = tf.placeholder(tf.int64, shape=[None], name="rel")
            self.training = tf.placeholder(tf.bool, name="training")
            self.labels = tf.placeholder(tf.float32, [None, self.NE],
                                         name='label')

    def _build_input_data(self):
        dataset = tf.data.TFRecordDataset("symptom.tfrecord",
                                          num_parallel_reads=8)

        # Create a description of the features.
        feature_description = {
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
        }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            return tf.io.parse_single_example(example_proto,
                                              feature_description)

        def _transform(d):
            # TODO: fixed this num_ent in tfrecord
            num_ent = 28754
            with tf.name_scope("transform"):
                zeros = tf.Variable(lambda: tf.zeros(num_ent), name="zeros")
                label = tf.convert_to_tensor(d["label"], dtype=tf.int64)
                d["label"] = tf.scatter_nd_update(
                    zeros, label, tf.ones_like(label, dtype=tf.float32))
                return d

        dataset = dataset.map(_parse_function).map(_transform)

        iterator = dataset.shuffle(30000).repeat(10).batch(4).prefetch(
            4).make_initializable_iterator()

        with tf.name_scope("batch_data"):
            self.batch_data = iterator.get_next("get_next")

    def _build_forward(self):
        ada = AdaE(self.NE, self.NR, self.C)
        self.logits = ada(self.e1, self.rel, self.training)

    def _build_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy_with_logits(
                labels=self.labels, logits=self.logits, label_smoothing=0.1),
                                      name="loss")

    def _build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0,
                                           dtype=tf.int32,
                                           trainable=False,
                                           name='global_step')
            self.lr = tf.placeholder_with_default(0.001, [], name='lr')
            self.optimize = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss, self.global_step, name="optimize")

    def _build_summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all(name="summary_op")

    def _build_train(self):
        self.init = tf.variables_initializer(tf.global_variables(),
                                             name='init')
        self.saver = tf.train.Saver(tf.global_variables())

    def export(self):
        self.build()

        definition = tf.Session().graph_def
        tf.io.write_graph(definition,
                          self.output_dir,
                          'model_txt.pb',
                          as_text=True)
        write_graph("AdaExport")


def export():
    NE = 22
    NR = 3
    C = 8
    ada_export = AdaExport(NE, NR, C)
    ada_export.export()


if __name__ == "__main__":
    # test()
    # tf.reset_default_graph()
    export()