#[macro_use]
extern crate log;
use anyhow::{anyhow, bail, Result};
use env_logger;
use platform::triple::Triple;
use serde_json;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;

fn main() -> Result<()> {
    println!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));
    env_logger::init();
    let mut heads: HashSet<String> = HashSet::new();
    let mut rels: HashSet<String> = HashSet::new();
    let mut tails: HashSet<String> = HashSet::new();
    let mut togs_head: HashSet<String> = HashSet::new();
    let mut togs_tail: HashSet<String> = HashSet::new();

    let filename = "data/chinese_symptom/all_result/triples.json";
    for line in BufReader::new(File::open(filename)?).lines() {
        if let Ok(line) = line {
            let triple: Triple = serde_json::from_str(&line)?;
            if triple.head.contains(",") {
                togs_head.insert(triple.head.to_string());
            }
            if triple.tail.contains(",") {
                togs_tail.insert(triple.tail.to_string());
            }
            heads.insert(triple.head.to_string());
            rels.insert(triple.rel.to_string());
            tails.insert(triple.tail.to_string());
        }
    }
    println!("heads: {:#?}", heads.len());
    println!("rels: {:#?}", rels.len());
    println!("tails: {:#?}", tails.len());
    println!("togs_head: {:#?}", togs_head.len());
    println!("togs_tail: {:#?}", togs_tail.len());
    // println!("{:#?}", rels);
    // write the togs to file
    let tog_head_file = "data/chinese_symptom/all_result/togs_head.txt";
    let tog_tail_file = "data/chinese_symptom/all_result/togs_tail.txt";
    let mut tog_head_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(tog_head_file)?;

    let mut tog_tail_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(tog_tail_file)?;

    for tog in togs_head.iter() {
        writeln!(tog_head_file, "{}", tog)?;
    }
    for tog in togs_tail.iter() {
        writeln!(tog_tail_file, "{}", tog)?;
    }

    Ok(())
}
