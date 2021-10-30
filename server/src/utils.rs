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
                        "[{}:{}] <{}> {} - {}",
                        record.file().unwrap_or("unknown"),
                        record.line().unwrap_or(0),
                        chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                        // buf.default_styled_level(record.level()),
                        buf.default_level_style(record.level()).set_bold(true).value(record.level()),
                        buf.style().set_intense(true).set_bold(true).set_color(env_logger::fmt::Color::Magenta).value(record.args())
                    )
                })
                .filter(None, log::LevelFilter::Debug)
                .init();
        }
    };
}
