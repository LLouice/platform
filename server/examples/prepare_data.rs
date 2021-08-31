use anyhow::Result;
use platform::triple::Triple;
use serde_json;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
};
#[macro_use]
extern crate log;
use env_logger;
// use hdf5;
use ndarray::prelude::*;
use ndarray::{stack, Array1, Array2};
use ndarray_stats::QuantileExt;

const FILES: [&'static str; 6] = [
    "症状相关检查",
    "症状相关疾病",
    "症状相关症状",
    "症状相关科室",
    "症状相关药品",
    "症状相关部位",
];

fn main() -> Result<()> {
    env_logger::init();
    // make_array()?;
    // write2hdf5()?;
    split_dataset()?;
    Ok(())
}

fn split_dataset() -> Result<()> {
    let array = make_array()?;
    let ents_head = array.slice(s![.., 0]).to_vec();
    let ents_tail = array.slice(s![.., 1]).to_vec();
    debug!("{:?}", ents_head.len());
    let mut ents_map = std::collections::HashMap::new();
    for e in ents_head.into_iter().chain(ents_tail.into_iter()) {
        *ents_map.entry(e).or_insert(0) += 1;
    }
    debug!("{:?}", ents_map.len());
    let ents_vec = {
        let mut ents_vec: Vec<_> = ents_map.into_iter().collect();
        ents_vec.sort_by(|a, b| a.1.cmp(&b.1).reverse());
        ents_vec
    };
    println!("{:?}", ents_vec.iter().take(50).collect::<Vec<_>>());
    Ok(())
}

fn write2hdf5() -> Result<()> {
    let array = make_array()?;
    let outfile = "data/chinese_symptom/all_result/triples.h5";
    let file = hdf5::File::create(outfile).expect("h5 file  not exist!");
    info!("hdf5 file: {:?}", file);

    let grp = file.create_group("all")?;
    let x = grp
        .new_dataset::<i64>()
        .create("x", array.shape().to_owned())?;

    x.write(array.view())?;
    info!("{:?}", x.shape());
    Ok(())
}

fn make_array() -> Result<Array2<i64>> {
    let filename = "data/chinese_symptom/all_result/triples_id.txt";
    info!("{}", &filename);
    let mut arrays = vec![];
    for line in BufReader::new(File::open(&filename)?).lines() {
        if let Ok(line) = line {
            let res: Result<Vec<i64>, _> =
                line.trim().split(",").map(|x| x.parse::<i64>()).collect();
            let res = res?;
            let array = Array1::from_vec(res);
            arrays.push(array);
        }
    }
    info!("{:?}", arrays.len());
    let res = {
        let tmp = arrays.iter().map(|x| x.view()).collect::<Vec<_>>();
        stack(Axis(0), tmp.as_slice())?
    };
    info!("{:?}", res.shape());
    info!("finish");
    Ok(res)
}

fn get_triples_id() -> Result<()> {
    let dir_name = "data/chinese_symptom/all_result";
    let outfile = "data/chinese_symptom/all_result/triples_id.txt";
    let mut outfile = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(outfile)?;

    let mut ent_map = HashMap::new();
    let mut rel_map = HashMap::new();

    for filename in &FILES {
        info!("{}", filename);
        let filename = format!("data/chinese_symptom/all_result/split/{}.txt", filename);
        for line in BufReader::new(File::open(&filename)?).lines() {
            if let Ok(line) = line {
                let triple: Triple = serde_json::from_str(&line)?;
                let Triple { head, rel, tail } = triple;
                let head_id = {
                    let len = ent_map.len();
                    *ent_map.entry(head).or_insert(len)
                };
                let tail_id = {
                    let len = ent_map.len();
                    *ent_map.entry(tail).or_insert(len)
                };
                let rel_id = {
                    let len = rel_map.len();
                    *rel_map.entry(rel).or_insert(len)
                };
                debug!("{}, {}, {}", head_id, rel_id, tail_id);
                writeln!(outfile, "{},{},{}", head_id, rel_id, tail_id)?;
            }
        }
    }
    // dump the map
    let ent_map_string = serde_json::to_string(&ent_map)?;
    std::fs::write(format!("{}/{}", dir_name, "ent_map.json"), ent_map_string)?;

    let rel_map_string = serde_json::to_string(&rel_map)?;
    std::fs::write(format!("{}/{}", dir_name, "rel_map.json"), rel_map_string)?;

    println!("ent_num: {:?}", ent_map.len());
    println!("rel_num: {:?}", rel_map.len());
    Ok(())
}
