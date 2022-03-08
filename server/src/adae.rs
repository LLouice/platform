use anyhow::Result;
use std::{io::BufReader, collections::HashMap, fs::File};

const NAME_ID_MAP: &str = "name_id_map.json";

fn load_map() -> Result<()> {
    let name_id_map: HashMap<String, usize> = {
        let reader = BufReader::new(File::open(NAME_ID_MAP)?);
        serde_json::from_reader(reader)?
    };
    println!("name_id_map len: {:?}", name_id_map.len());
    Ok(())
}

mod tests {
    use super::*;

    #[test]
    fn test_load_map() {
        load_map();
    }
}
