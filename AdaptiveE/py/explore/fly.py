import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_gpu, write_graph
import tensorflow as tf



def build():
    pass



def main():
    set_gpu(1)


    with tf.variable_scope("ab"):
        a = tf.Variable(1, name="a")
        b = tf.Variable(2, name="b")
        c = tf.add(a, b, name="c")
        c2 = tf.add(a, b, name="c2")

    e = tf.Variable(1, name="e")
    f = tf.Variable(2, name="f")


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(10):
            sess.run(c)
            sess.run(tf.add(a, b, name=f"i_{i}"))
            d = tf.add(a, b, name="d") # new op with name d_i
            sess.run(d)

            sess.run(tf.add(e, f, name=f"j_{i}"))
            h = tf.add(e, f, name="h")
            sess.run(h)


    write_graph("fly")



if __name__ == '__main__':
    main()
