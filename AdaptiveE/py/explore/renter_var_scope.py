import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_gpu, write_graph
import tensorflow as tf


def build_vallina():
    with tf.variable_scope("first", auxiliary_name_scope=False) as scope:
        # with tf.name_scope("first/"):
        with tf.name_scope(scope.original_name_scope):
            a = tf.get_variable(name="a", initializer=tf.constant(1.))
            b = tf.get_variable(name="b", initializer=tf.constant(2.))

            with tf.variable_scope("second",
                                   auxiliary_name_scope=False) as scope:
                # with tf.name_scope("second/"):
                with tf.name_scope(scope.original_name_scope):
                    a2 = tf.get_variable(name="a2",
                                         initializer=tf.constant(1.))

    print(a)
    print(b)
    print(a2)

    # with tf.variable_scope("first", reuse=tf.AUTO_REUSE):
    #     a = tf.get_variable(name="a", initializer=tf.constant(1.))
    #     c = tf.get_variable(name="c", initializer=tf.constant(3.))
    #     d = tf.Variable(tf.constant(4.), name="d")

    with tf.variable_scope("first",
                           reuse=tf.AUTO_REUSE,
                           auxiliary_name_scope=False) as scope:
        # with tf.name_scope("first/"):
        print(scope.original_name_scope)
        with tf.name_scope(scope.original_name_scope):
            a = tf.get_variable(name="a", initializer=tf.constant(1.))
            c = tf.get_variable(name="c", initializer=tf.constant(3.))
            d = tf.Variable(tf.constant(4.), name="d")

            with tf.variable_scope("second",
                                   auxiliary_name_scope=False) as scope:
                # with tf.name_scope("second/"):
                print(scope.original_name_scope)
                with tf.name_scope(scope.original_name_scope):
                    a2 = tf.get_variable(name="a2",
                                         initializer=tf.constant(1.))
                    d2 = tf.Variable(tf.constant(4.), name="d2")

    print(a)
    print(c)
    print(d)
    print(a2)
    print(d2)

    with tf.variable_scope("first", reuse=True,
                           auxiliary_name_scope=False) as scope:
        with tf.name_scope("first/"):
            with tf.variable_scope("second",
                                   auxiliary_name_scope=False) as scope:
                with tf.name_scope("second/"):
                    a2 = tf.get_variable(name="a2",
                                         initializer=tf.constant(1.))
    print(a2)

    # with tf.variable_scope("first", reuse=tf.AUTO_REUSE, auxiliary_name_scope=False) as scope:
    #     with tf.name_scope(scope.original_name_scope):
    #         a = tf.get_variable(name="a", initializer=tf.constant(1.))
    #         c = tf.get_variable(name="c", initializer=tf.constant(3.))
    #         e = tf.Variable(tf.constant(5.), name="e")

    # print(a)
    # print(c)
    # print(e)

    # with tf.variable_scope("first/", reuse=tf.AUTO_REUSE):
    #     a = tf.get_variable(name="a", initializer=tf.constant(1.))
    #     c = tf.get_variable(name="c", initializer=tf.constant(3.))


def get_scope(name, reuse=None):
    if not reuse:
        scope = tf.variable_scope(name, reuse=reuse)
        return tf.variable_scope(""), scope
    else:
        scope_0 = tf.variable_scope(name,
                                    reuse=reuse,
                                    auxiliary_name_scope=False)
        scope = tf.name_scope(None)
        return scope_0, scope


def get_scope_plus(name, reuse=None):
    if not reuse:
        scope = tf.variable_scope(name, reuse=reuse)
        return tf.variable_scope(""), scope
    else:
        scope_0 = tf.variable_scope(name,
                                    reuse=reuse,
                                    auxiliary_name_scope=False)
        scope = tf.name_scope(None)
        return scope_0, scope


def build():
    scope_0, scope = get_scope("first")
    with scope_0:
        with scope:
            a = tf.get_variable(name="a", initializer=tf.constant(1.))
            # b = build_sub(reuse=None)

    print(a)
    # print(b)

    scope_0, scope = get_scope("first", reuse=True)
    with scope_0:
        with scope:
            a = tf.get_variable(name="a", initializer=tf.constant(1.))
            e = tf.Variable(tf.constant(5.), name="e")
    #         b = build_sub(reuse=True)

    print(a)
    print(e)
    # print(b)


def build_sub(reuse=None):
    scope_0, scope = get_scope("second", reuse=reuse)
    print(scope_0, scope)
    with scope_0:
        with scope:
            b = tf.get_variable(name="b", initializer=tf.constant(1.))
    return b


def get_scope2(prefix, name, reuse=None):
    prefix = f"{prefix}{name}/"
    if not reuse:
        scope1 = tf.variable_scope(name, auxiliary_name_scope=False)
    else:
        scope1 = tf.variable_scope(name,
                                   reuse=True,
                                   auxiliary_name_scope=False)
    scope2 = tf.name_scope(prefix)
    return prefix, scope1, scope2


class Scope:
    def __init__(self, name, prefix="", reuse=None):
        self.prefix = f"{prefix}{name}/"
        if not reuse:
            self.scope1 = tf.variable_scope(name, auxiliary_name_scope=False)
        else:
            self.scope1 = tf.variable_scope(name,
                                            reuse=True,
                                            auxiliary_name_scope=False)
        self.scope2 = tf.name_scope(self.prefix)

    def __enter__(self):
        self.scope1.__enter__()
        self.scope2.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.scope2.__exit__(exc_type, exc_value, traceback)
        self.scope1.__exit__(exc_type, exc_value, traceback)
        return True


def build_plain():
    with Scope("first") as scope:
        a = tf.get_variable(name="a", initializer=tf.constant(1.))
        d = tf.Variable(tf.constant(5.), name="d")

        print(a)
        print(d)
        print(scope.prefix)

        with Scope("second", scope.prefix) as scope2:
            b = tf.get_variable(name="b", initializer=tf.constant(1.))
            e = tf.Variable(tf.constant(5.), name="e")

        print(b)
        print(e)
        print(scope2.prefix)

    with Scope("first", reuse=True) as scope:
        a = tf.get_variable(name="a", initializer=tf.constant(1.))
        f = tf.Variable(tf.constant(5.), name="f")

        print(a)
        print(f)
        print(scope.prefix)

        with Scope("second", scope.prefix, True) as scope2:
            b = tf.get_variable(name="b", initializer=tf.constant(1.))
            bb = tf.get_variable(name="bb", initializer=tf.constant(1.))
            g = tf.Variable(tf.constant(5.), name="g")

        print(b)
        print(bb)
        print(g)
        print(scope2.prefix)

    # prefix, scope = get_scope2("", "first", reuse=True)
    # with scope:
    #     a = tf.get_variable(name="a", initializer=tf.constant(1.))
    #     e = tf.Variable(tf.constant(5.), name="e")
    #     build_plain_sub(prefix, True)
    #     # with tf.name_scope("under_slash"):
    #     #     s = tf.Variable(tf.constant(5.), name="s")

    #     #     with tf.name_scope("third"):
    #     #         ss = tf.Variable(tf.constant(5.), name="ss")

    # print(a)
    # print(e)


def build_plain_sub(prefix, reuse=None):
    if not reuse:
        prefix, scope = get_scope2(prefix, "second", None)
        with scope:
            b = tf.get_variable(name="b", initializer=tf.constant(1.))
        print(b)

        return prefix
    else:
        prefix, scope = get_scope2(prefix, "second", reuse=True)
        with scope:
            b = tf.get_variable(name="b", initializer=tf.constant(1.))
            f = tf.Variable(tf.constant(5.), name="f")
        print(b)
        print(f)
        return prefix


def main():
    set_gpu(3)
    # build()
    build_plain()

    write_graph("renter_var_scope")


if __name__ == '__main__':
    main()
