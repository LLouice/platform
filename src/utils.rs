use chrono::Local;

pub fn get_current_timestamp() -> String {
    let dt = Local::now();
    dt.format("%Y-%m-%dT%H:%M:%S").to_string()
}
