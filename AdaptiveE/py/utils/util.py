import tensorflow as tf

def write_graph(name):
    writer = tf.summary.FileWriter(f"./graphs/{name}", tf.get_default_graph())
    writer.close()
