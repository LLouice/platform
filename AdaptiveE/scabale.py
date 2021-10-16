import tensorflow as tf
from utils import set_gpu
from utils import embedding as utils_embedding

# no upstream function for test
class ScalabelGCN(object):
    '''
    reuse: for create may ScalabelGCN instances, shared or individual
    '''
    def __init__(self, bs, max_id, dim=32, v_dim=10, num_layers=2, store_learning_rate=0.001, store_init_maxval=0.05, use_residual=False, reuse=None):
        super().__init__()
        # specific batch_size for reuse related variable
        self.bs = bs
        self.max_id = max_id
        self.dim = dim
        self.v_dim = v_dim
        self.num_layers = num_layers
        self.max_id = max_id
        self.store_learning_rate = store_learning_rate
        self.store_init_maxval = store_init_maxval
        self.use_residual = use_residual

        self.reuse = reuse
        self.build()

    def build(self):
        with tf.variable_scope("ScalabelGCN", reuse=self.reuse):
            with tf.name_scope("input"):
                self.node = tf.placeholder("node", shape=[self.bs], dtype=tf.int64)
                self.neigh = tf.sparse_placeholder("neigh", shape=[self.bs, self.max_id] ,dtype=tf.float32)

            self.emb = tf.get_variable('emb',
                                            shape=[self.bs, self.dim],
                                            initializer=tf.glorot_uniform_initializer())

        # for model the relation about the head and relationship combined embedding(bs, 2E)
        with tf.variable_scope('ada_vector'):
            self.v1 = tf.get_variable('v1',
                                            shape=[2*self.dim, self.v_dim],
                                            initializer=tf.glorot_normal_initializer())

            self.v2 = tf.get_variable('v2',
                                            shape=[self.v_dim, 2*self.dim],
                                            initializer=tf.glorot_normal_initializer())

            with tf.variable_scope("cache"):
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

    def call(self, training=None):
        if not training:
            pass


        # ada vector
        # [2E, V] x [V, 2E] -> [2E, 2E] (ada_adj A)
        # A + A_raw_partial(sparse_neigh) TODO pre-create this batch sparse_neigh
        # agg: A x EM (embedding matrix) <=> [bs, N] [N, E] -> [bs, E]

        with tf.variable_scope("ScalabelGCN", reuse=self.reuse):
        # (nodes, node_embedding, neigh, sparse_neigh) = inputs
            # [bs] -> [bs, E]

            node_embedding = tf.nn.embedding_lookup(self.emb, self.node, name="node_embedding")
            neigh_embedding = tf.nn.embedding_lookup(self.emb, self.neigh, name="neigh_embedding")

            node_embeddings = []
            neigh_embeddings = []

            for layer in range(self.num_layers):
                # [b, E] [b, N]
                agg_embedding = tf.sparse_dense_matmul(node_embedding, neigh, name="agg_embedding")
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
