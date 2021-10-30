import sys
import tensorflow as tf
from utils import set_gpu



# def set_gpu(id):
#     gpus = tf.config.experimental.list_physical_devices("GPU")

#     try:
#         tf.config.experimental.set_memory_growth(gpus[id], True)
#         tf.config.experimental.set_visible_devices(gpus[id], "GPU")
#         logical_gpus = tf.config.experimental.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except Exception as e:
#         print(f"Can't not use gpu:{id}", file=sys.stderr)
#         print(e, file=sys.stderr)


def main():
    set_gpu(2)

    a = tf.constant(1, dtype=tf.float32, name="a");
    b = tf.constant(3, dtype=tf.float32, name="b");
    c = tf.add(a, b, name="c");


    writer = tf.summary.FileWriter('./graphs/demo', tf.get_default_graph())

    print(c)
    print(tf.get_default_graph().as_graph_def())

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(c));

    writer.close()


if __name__ == "__main__":
    main()
