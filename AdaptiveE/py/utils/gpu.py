import sys
import tensorflow as tf


def set_gpu(id):
    gpus = tf.config.experimental.list_physical_devices("GPU")

    try:
        tf.config.experimental.set_memory_growth(gpus[id], True)
        tf.config.experimental.set_visible_devices(gpus[id], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except Exception as e:
        print(f"Can't not use gpu:{id}", file=sys.stderr)
        print(e, file=sys.stderr)
