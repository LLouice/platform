use serde_json;
use std::collections::HashMap;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// id
    #[structopt(short = "i", long)]
    id: Option<usize>,

    /// ent
    #[structopt(short = "e", long)]
    ent: Option<String>,
}

fn main() {
    let opt = Opt::from_args();

    let filename = "data/chinese_symptom/all_result/ent_map.json";
    let ent_map: HashMap<String, i32> =
        serde_json::from_slice(std::fs::read(filename).unwrap().as_slice()).unwrap();
    //     serde_json::from_str(std::fs::read_to_string(filename).unwrap().as_str()).unwrap();
    let ent_map_rev: HashMap<i32, String> = ent_map
        .iter()
        .map(|(k, v)| (v.clone(), k.clone()))
        .collect();
    if let Some(id) = opt.id {
        println!("id: {:?}", ent_map_rev.get(&(id as i32)));
    };
    if let Some(ent) = opt.ent {
        println!("id: {:?}", ent_map.get(&ent));
    };
}
