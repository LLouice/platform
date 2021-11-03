import tensorflow as tf


class Scope:
    def __init__(self, name, prefix="", reuse=None):
        self.prefix = f"{prefix}{name}/"

        self.scope1 = tf.variable_scope(name,
                                        reuse=reuse,
                                        auxiliary_name_scope=False)
        self.scope2 = tf.name_scope(self.prefix)

    def __enter__(self):
        self.scope1.__enter__()
        print("scope1:: ", self.scope1.original_name_scope)
        self.scope2.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.scope2.__exit__(exc_type, exc_value, traceback)
        self.scope1.__exit__(exc_type, exc_value, traceback)
        return True
