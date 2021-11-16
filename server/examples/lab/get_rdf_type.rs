#[macro_use]
extern crate log;
use anyhow::{anyhow, bail, Result};
use env_logger;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

fn main() -> Result<()> {
    println!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));
    env_logger::init();
    let mut type_s = HashSet::new();
    let mut type_instance_s = HashSet::new();

    let filename = "data/chinese_symptom/not_valid.txt";
    for line in BufReader::new(File::open(filename)?).lines() {
        if let Ok(line) = line {
            let mut line: Vec<&str> = line.split("\t").collect();

            let (tail, rel, head) = (
                line.pop().unwrap().trim().to_string(),
                line.pop().unwrap().trim().to_string(),
                line.pop().unwrap().trim().to_string(),
            );
            type_s.insert(rel);
            type_instance_s.insert(tail);
        }
    }
    println!("{:#?}", type_s.len());
    println!("{:#?}", type_s);
    println!("{:#?}", type_instance_s.len());

    Ok(())
}
