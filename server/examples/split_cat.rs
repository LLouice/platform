#[macro_use]
extern crate log;
use anyhow::{anyhow, bail, Result};
use env_logger;
use platform::triple::Triple;
use seq_macro::seq;
use serde_json;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::PathBuf;

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

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Activate debug mode
    // short and long flags (-d, --debug) will be deduced from the field's name
    #[structopt(short, long)]
    debug: bool,

    /// head
    #[structopt(short = "h", long)]
    head: Option<String>,

    /// tail
    #[structopt(short = "t", long)]
    tail: Option<String>,
}

fn main() -> Result<()> {
    println!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));
    env_logger::init();
    // let cats = ["疾病", "症状", "检查", "部位", "药品", "科室"];

    let opt = Opt::from_args();
    println!("{:?}", opt);
    let (ent, is_head) = match (&opt.head, &opt.tail) {
        (Some(ent), None) => (ent, true),
        (None, Some(ent)) => (ent, false),
        _ => unreachable!(),
    };

    let mut S = HashSet::new();
    let mut linum = 0;
    seq!(N in 1..=17 {
        if (is_head && REL#N.starts_with(ent)) || (!is_head && REL#N.ends_with(ent)) {
            println!("{}", REL#N);
            let filename = format!("data/chinese_symptom/all_result/split/{}.txt", REL#N);
            for line in BufReader::new(File::open(filename)?).lines() {
                linum += 1;
                if let Ok(line) = line {
                    let triple: Triple = serde_json::from_str(&line)?;
                    if is_head {
                        S.insert(triple.head.to_string());
                    }else{
                        S.insert(triple.tail.to_string());
                    }
                }
            }
        }
    });

    let outfile = "data/chinese_symptom/all_result/cats";
    let outfile = format!("{}/{}.txt", outfile, ent);
    let mut outfile = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(outfile)?;

    println!("all: {}", linum);
    println!("unique: {}", S.len());

    for item in S.into_iter() {
        writeln!(outfile, "{}", item);
    }

    Ok(())
}
