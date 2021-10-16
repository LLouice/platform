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



    # init = [v.initializer for v in tf.global_variables() if "c" not in v.name]

    # print(init)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(3):
            for i in range(10):
               if i == 0 and e == 0:
                        with tf.variable_scope("c"):
                            c = tf.get_variable(name="c", initializer=tf.constant(0.))
                            c1 = tf.get_variable(name="c1", initializer=tf.constant(0.))
                            c2 = tf.get_variable(name="c2", initializer=tf.constant(0.))
                        sess.run([c.initializer, c1.initializer, c2.initializer])
               else:
                    # use "/" or auxiliary_name_scope=False? name_scope is useless for get_variable
                    with tf.variable_scope("c", reuse=True):
                        c = tf.get_variable(name="c", initializer=tf.constant(0.))
                        c1 = tf.get_variable(name="c1", initializer=tf.constant(0.))
                        c2 = tf.get_variable(name="c2", initializer=tf.constant(0.))
               print(sess.run([c, c1, c2]))

    write_graph("fly_var_scope")



if __name__ == '__main__':
    main()
