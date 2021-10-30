import tensorflow as tf


def build_config_proto(id):
    device_count = {'GPU': id}
    config=tf.ConfigProto(device_count=device_count)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    return config.SerializeToString()
