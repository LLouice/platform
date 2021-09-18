#![allow(dead_code)]

use chrono::NaiveDateTime;
// use futures::stream::{FuturesUnordered, StreamExt};
use neo4rs::*;
use tokio;

use platform::kg::Kg;
use platform::neo4j::init_graph;

#[tokio::main]
async fn main() {
    stat().await;
}

// stats
async fn node_num() {
    let graph = init_graph().await;
    let cypher = format!("match (ps:{}) return count(ps) as num", "Symptom");
    // let cypher = format!("match (ps:{}{{name: \"肩背痛\"}}) return ps", "Symptom");
    let mut result = graph.execute(query(cypher.as_str())).await.unwrap();

    while let Ok(Some(row)) = result.next().await {
        // let node: Node = row.get("pt").unwrap();
        // let labels = node.labels();
        // let name: String = node.get("name").unwrap();
        // let updated_time: NaiveDateTime = node.get("updated_time").unwrap();
        // res.push(LinksResult::new(name, labels, updated_time));
        println!("{:#?}", row);
        println!("{:?}", row.get::<i64>("num"));
    }
}

async fn stat() {
    let stats = Kg::stat().await;
    println!("{:?}", stats);
}
