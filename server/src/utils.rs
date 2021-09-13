//! utils function and macros


pub fn get_current_timestamp() -> String {
    let dt = chrono::Local::now();
    dt.format("%Y-%m-%dT%H:%M:%S").to_string()
}

/// init env_logger once
/// src/bin/logger_example.rs:20 2020-11-30T19:34:16 [INFO] - hello world
#[macro_export]
macro_rules! init_env_logger {
    () => { {
        use std::io::Write;

        env_logger::Builder::new()
                .format(|buf, record| {
                    writeln!(
                        buf,
                        "{}:{} {} [{}] - {}",
                        record.file().unwrap_or("unknown"),
                        record.line().unwrap_or(0),
                        chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                        record.level(),
                        record.args()
                    )
                })
                .filter(None, log::LevelFilter::Debug)
                .init();
        }
    };
}
