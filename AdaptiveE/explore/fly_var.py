import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_gpu, write_graph
import tensorflow as tf



def build():
    pass



def main():
    set_gpu(3)


    with tf.variable_scope("ab"):
        a = tf.get_variable(name="a", initializer=tf.constant(1.))
        b = tf.get_variable(name="b", initializer=tf.constant(2.))

    # pre-create not initialized variable
    with tf.variable_scope("c"):
        c = tf.get_variable(name="c", initializer=tf.constant(0.))
        c1 = tf.get_variable(name="c1", initializer=tf.constant(0.))
        c2 = tf.get_variable(name="c2", initializer=tf.constant(0.))


    init = [v.initializer for v in tf.global_variables() if "c" not in v.name]

    print(init)

    with tf.Session() as sess:
        sess.run(init)

        for e in range(3):

            for i in range(10):
                if e == 0:
                    if i == 0:
                        sess.run(c.initializer)

                    if i == 1:
                        sess.run(c1.initializer)

                    if i == 2:
                        sess.run(c2.initializer)

                sess.run(c)

                if i > 0:
                    sess.run(c1)

                if i > 1:
                    sess.run(c2)

    write_graph("fly_var")



if __name__ == '__main__':
    main()
