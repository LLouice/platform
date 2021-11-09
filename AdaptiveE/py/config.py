import tensorflow as tf


def build_config_proto(visible_device_list: str = "0",
                       log_device_placement: bool = False) -> tf.ConfigProto:
    # device_count = {'GPU': id}
    # config=tf.ConfigProto(device_count=device_count)
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    config.gpu_options.visible_device_list = visible_device_list
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    return config
