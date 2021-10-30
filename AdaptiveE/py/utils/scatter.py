import tensorflow as tf

def scatter_update_tensor(x, indices, updates):
    '''
    Utility function similar to `tf.scatter_update`, but performing on Tensor
    scatter_update(x, [[2],[3]], [22, 33]) => [., ., 22, 33, ...]
    scatter_nd_update(x, [[2, 3]], [22, 33])
    =>  [
            [                 ]
            [                 ]
            [., ., 22,  ...   ]
            [                 ]
        ]
    '''
    x_shape = tf.shape(x)
    patch = tf.scatter_nd(indices, updates, x_shape)
    mask = tf.greater(tf.scatter_nd(indices, tf.ones_like(updates), x_shape), 0)
    return tf.where(mask, patch, x)
