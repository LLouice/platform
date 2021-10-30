import tensorflow as tf
from utils import set_gpu

set_gpu(3)

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30

with tf.name_scope("input"):
    X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
    Y = tf.placeholder(tf.int32, [batch_size, 10], name='label')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
with tf.name_scope("model"):
    w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
    b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

    # Step 4: build model
    # the model that returns the logits.
    # this logits will be later passed through softmax layer
    logits = tf.matmul(X, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
with tf.name_scope("loss"):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
    loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch
# loss = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(logits) * tf.log(Y), reduction_indices=[1]))

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
with tf.name_scope("accuracy"):
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
