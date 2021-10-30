import tensorflow as tf
from utils import set_gpu
from utils import embedding as utils_embedding


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
        use_seg_emb=False,
        order=2,
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
        self.use_seg_emb = use_seg_emb
        self.order = order

        self.build()




    def build(self):
        with tf.variable_scope('emb'):
            self.emb_e = tf.get_variable('emb_e',
                                            shape=[self.NE, self.E],
                                            initializer=tf.glorot_uniform_initializer())


            self.emb_rel = tf.get_variable('emb_r',
                                            shape=[self.NR, self.E],
                                            initializer=tf.glorot_uniform_initializer())



        # TODO: print the graph, inspect the shape
        # for model the entity-relation combined inner relationship
        with tf.variable_scope('ada_vector'):
            self.v1 = tf.get_variable('v1',
                                            shape=[2*self.E, self.v_dim],
                                            initializer=tf.glorot_normal_initializer())

            self.v2 = tf.get_variable('v2',
                                            shape=[self.v_dim, 2*self.E],
                                            initializer=tf.glorot_normal_initializer())


        # "GCN"
        self.gcn = GCN(order, C)


        # optional: increase the embedding
        # [b, E] - reshape -> [b, E, 1] (for torch bn) - fc? -> [b, E, c]
        # "IC"
        with tf.name_scope('ic_e1'):
            self.bn_e1 = tf.layers.BatchNormalization(axis=-1)
            self.dp_e1 = tf.layers.Dropout(rate=self.inp_dp)

        with tf.name_scope('ic_rel'):
            self.bn_rel = tf.layers.BatchNormalization(axis=-1)
            self.dp_rel = tf.layers.Dropout(rate=self.inp_dp)

        with tf.name_scope('ic_hid'):
            self.bn_hid = tf.layers.BatchNormalization(axis=-1)
            self.dp_hid = tf.layers.Dropout(rate=self.hid_dp)

        with tf.name_scope('ic_last'):
            self.bn_last = tf.layers.BatchNormalization(axis=-1)
            self.dp_last = tf.layers.Dropout(rate=self.last_dp)

        with tf.name_scope('fc'):
            # 2*E*C -> E
            self.fc = tf.layers.Dense(self.E, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.glorot_normal_initializer(),
    bias_initializer=tf.zeros_initializer())


         self.b = tf.get_variable('b', shape=[self.E], initializer=tf.zero_initializer())

    def call():
        pass



class ScalabelGCN(object):
    def __init__(self, max_id, dim=32, num_layers=2, store_learning_rate=0.001, store_init_maxval=0.05):
        super().__init__()
        self.max_id = max_id
        self.dim = dim
        self.num_layers = num_layers
        self.max_id = max_id
        self.store_learning_rate = store_learning_rate
        self.store_init_maxval = store_init_maxval

        self.build()

    def build(self):
        with tf.variable_scope("ScalabelGCN") as scope:
            self.stores = [
                tf.get_variable('store_layer_{}'.format(i),
                                [self.max_id + 2, self.dim],
                                initializer=tf.random_uniform_initializer(
                                    maxval=self.store_init_maxval, seed=1),
                                trainable=False,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
                for i in range(1, self.num_layers)]
            self.gradient_stores = [
                tf.get_variable('gradient_store_layer_{}'.format(i),
                                [self.max_id + 2, self.dim],
                                initializer=tf.zeros_initializer(),
                                trainable=False,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
                for i in range(1, self.num_layers)]
            self.store_optimizer = tf.train.AdamOptimizer(self.store_learning_rate)


    def call(self, inputs, training=None):
        '''
        TODO: add raw adj
        inputs: (nodes, nodes_embedding, neigh, sparse_neigh) :([b, E], [b, N], [b, ..])
        '''

        if not training:
            # return super(ScalableGCNEncoder, self).call(inputs)
            pass


        (nodes, node_embedding, neigh, sparse_neigh) = inputs




        node_embeddings = []
        neigh_embeddings = []

        for layer in range(self.num_layers):

            # neigh = neigh + sparse_neigh

            # TODO: pre-build this op
            agg_embedding = tf.matmul(node_embedding, neigh)
            if self.use_residual:
                # Can use node_embedding name again
                node_embedding_new = tf.add(node_embedding, agg_embedding)
            else:
                node_embedding_new = agg_embedding;

            node_embeddings.append(node_embedding_new)

            if layer < self.num_layers - 1:
                neigh_embedding = \
                    tf.nn.embedding_lookup(self.stores[layer], neigh)
                neigh_embeddings.append(neigh_embedding)

        self.update_store_op = self._update_store(nodes, node_embeddings)

        store_loss, self.optimize_store_op = \
            self._optimize_store(node, node_embeddings)
        self.get_update_gradient_op = lambda loss: \
            self._update_gradient(loss + store_loss,
                                  neighbor,
                                  neigh_embeddings)

        output_shape = inputs.shape.concatenate(node_embedding.shape[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(node_embedding, output_shape)

    def _update_store(self, node, node_embeddings):
        update_ops = []
        for store, node_embedding in zip(self.stores, node_embeddings):
            update_ops.append(
                utils_embedding.embedding_update(store, node, node_embedding))
        return tf.group(*update_ops)

    def _update_gradient(self, loss, neighbor, neigh_embeddings):
        update_ops = []
        for gradient_store, neigh_embedding in zip(
              self.gradient_stores, neigh_embeddings):
            embedding_gradient = tf.gradients(loss, neigh_embedding)[0]
            update_ops.append(
                utils_embedding.embedding_add(gradient_store,
                                              neighbor, embedding_gradient))
        return tf.group(*update_ops)

    def _optimize_store(self, node, node_embeddings):
        if not self.gradient_stores:
            return tf.zeros([]), tf.no_op()

        losses = []
        clear_ops = []
        for gradient_store, node_embedding in zip(
              self.gradient_stores, node_embeddings):
            embedding_gradient = tf.nn.embedding_lookup(gradient_store, node)
            with tf.control_dependencies([embedding_gradient]):
                clear_ops.append(
                    utils_embedding.embedding_update(
                        gradient_store, node,
                        tf.zeros_like(embedding_gradient)))
            losses.append(tf.reduce_sum(node_embedding * embedding_gradient))

        store_loss = tf.add_n(losses)
        with tf.control_dependencies(clear_ops):
            return store_loss, self.store_optimizer.minimize(store_loss)
