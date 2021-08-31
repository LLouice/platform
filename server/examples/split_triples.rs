#[macro_use]
extern crate log;
use anyhow::{anyhow, bail, Result};
use env_logger;
use platform::triple::Triple;
use seq_macro::seq;
use serde_json;
use std::fs::{File, OpenOptions};
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;

const REL1: &str = "检查相关症状";
const REL2: &str = "疾病相关疾病";
const REL3: &str = "疾病相关症状";
const REL4: &str = "检查相关部位";
const REL5: &str = "症状相关部位";
const REL6: &str = "症状相关症状";
const REL7: &str = "症状相关科室";
const REL8: &str = "症状相关疾病";
const REL9: &str = "症状相关检查";
const REL10: &str = "疾病相关检查";
const REL11: &str = "疾病相关部位";
const REL12: &str = "疾病相关药品";
const REL13: &str = "检查相关检查";
const REL14: &str = "检查相关疾病";
const REL15: &str = "疾病相关科室";
const REL16: &str = "症状相关药品";
const REL17: &str = "检查相关科室";

fn main() -> Result<()> {
    println!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));
    env_logger::init();

    let filename = "data/chinese_symptom/all_result/triples.json";
    let outfile = "data/chinese_symptom/all_result/split/";
    let mut outfiles = vec![];
    seq!(N in 1..=17 {
        let f = format!("{}/{}.txt", outfile, REL#N);
        outfiles.push(OpenOptions::new().create(true).write(true).truncate(true).open(f)?);
    });

    let mut linum = 0;
    for line in BufReader::new(File::open(filename)?).lines() {
        linum += 1;
        if let Ok(line) = line {
            let triple: Triple = serde_json::from_str(&line)?;

            seq!(N in 1..=17 {
                if triple.rel.as_str() == REL#N  {
                    // println!("{} => {:?}", REL#N, triple);
                    writeln!(outfiles[N-1], "{}", line)?;
                }
            });
        }
        // if linum > 10 {
        //     break;
        // }
    }

    Ok(())
}
