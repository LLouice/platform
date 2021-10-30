
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_gpu, write_graph
import tensorflow as tf



count = tf.get_variable("count", initializer=tf.constant([0.]))


def cond(i):
    return tf.less(i, 5)

def loop_body(i):
    # this op is not execute!!
    update_op =tf.scatter_update(count, [0], tf.cast(i, tf.float32))
    with tf.control_dependencies([update_op]):
        return tf.add(i, 1)



def build():
    i = tf.while_loop(cond, loop_body, [tf.constant(0)])
    write_graph("while_loop_var")
    return i

def main():
    global count
    set_gpu(2)
    i = build()
    with tf.control_dependencies([i]):
        c2 = count.read_value()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # i = sess.run([i])
        # print(i)
        # count = sess.run(count)
        # print(count)
        print(sess.run(c2))


if __name__ == "__main__":
    main()
