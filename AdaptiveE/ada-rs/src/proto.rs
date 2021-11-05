use anyhow::Result;
use tensorflow_proto::tensorflow::{self, ConfigProto};
use std::convert::TryFrom;

pub fn build_config_proto(visible_device_list: String, log_device_placement: bool) -> Result<Vec<u8>> {
    let config_proto = ConfigProto {
        log_device_placement,
        gpu_options: Some(tensorflow::GpuOptions {
            allow_growth: true,
            visible_device_list,
            per_process_gpu_memory_fraction: 0.9,
            ..Default::default()
        }),
        ..Default::default()
    };
    Ok(Vec::try_from(&config_proto)?)
}
