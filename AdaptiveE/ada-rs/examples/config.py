import tensorflow as tf


def build_config_proto(id):
    # device_count = {'GPU': id}
    # config=tf.ConfigProto(device_count=device_count)
    config=tf.ConfigProto(log_device_placement=False)
    config.gpu_options.visible_device_list= str(id) #only see the gpu id
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    return config.SerializeToString()


if __name__ == "__main__":
    config_bytes = build_config_proto(0)
    print(config_bytes)
    with open("config.proto", "wb") as f:
        f.write(config_bytes)
