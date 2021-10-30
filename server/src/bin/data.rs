use anyhow::{anyhow, bail, Result};
use chrono::NaiveDateTime;
use clap::{App, Arg};
use neo4rs::*;
use strum::IntoEnumIterator;

use std::collections::{BTreeSet, HashMap};
use std::fs::OpenOptions;
use std::path::PathBuf;

use platform::data::NodeLabel;
use platform::init_env_logger;
use platform::neo4j::init_graph;

#[tokio::main]
async fn main() {
    init_env_logger!();

    let mut app = App::new("kg")
        .version("1.0")
        .author("LLouice")
        .about("knowledge graph CLI")
        .license("MIT OR Apache-2.0")
        .arg(
            Arg::new("query_dup")
                .short('q')
                .long("query_dup")
                .about("query duplicate nodes"),
        )
        .arg(
            Arg::new("delete_dup")
                .short('d')
                .long("delete_dup")
                .about("delete nodes"),
        )
        .arg(
            Arg::new("write")
                .short('w')
                .about("export graph to text file"),
        )
        .arg(
            Arg::new("file")
                .short('f')
                .long("file")
                .default_value("graph.txt")
                .about("export graph to text file"),
        );
    let mut help = Vec::new();
    app.write_help(&mut help).expect("error on write app help");
    let help = String::from_utf8(help).expect("failed to convert help Vec<u8> to String");
    let matches = app.get_matches();

    let res;

    // let res = query_all_data().await;
    if matches.is_present("query_dup") {
        res = query_dup().await;
    } else if matches.is_present("delete_dup") {
        res = delete_dup().await;
    } else if matches.is_present("write") {
        if let Some(file) = matches.value_of("file") {
            res = graph2file(file).await;
        } else {
            res = Ok(())
        }
    } else if let Some(file) = matches.value_of("file") {
        res = graph2file(file).await;
    } else {
        res = Ok(());
        println!("{}", help);
    }

    // let res = quick_query().await;
    if let Err(e) = res {
        log::error!("{:?}", e);
    } else {
        println!("{:?}", res);
    }
    eprintln!("main end");
}

// section query all links
async fn query_all_data() -> Result<()> {
    eprintln!("query all data ...");
    let graph = init_graph().await;
    let label = "Symptom";
    let cypher_health = "MATCH(node:Symptom) return node limit 10";
    let cypher_out = format!(
        // "MATCH(node:{}) -[r]-> (t)  RETURN node, DISTINCT t limit 10",
        "MATCH(node:{}) -[r]-> (t)  RETURN node, t limit 10",
        label,
    );
    /*
    let cypher_in = format!(
        "MATCH(h) -[r]-> (node:{}) WHERE rand() < 0.5 RETURN node, COUNT(DISTINCT h) as num LIMIT {}",
        label,
        limit
    );
     */
    let cypher = cypher_out;
    let cypher = cypher_health;
    log::debug!("{}", cypher);
    let mut result = graph
        .execute(query(cypher.as_ref()))
        .await
        .map_err(|_| anyhow!("error on execute cypher"))?;

    // get query result
    // let mut res = vec![];
    while let Ok(Some(row)) = result.next().await {
        log::debug!("row is: {:#?}\n\n", row);

        let node: Node = row
            .get("node")
            .ok_or_else(|| anyhow!("no key `node` in row"))?;
        let name: String = node
            .get("name")
            .ok_or_else(|| anyhow!("no key `name` in node"))?;
        // redundant info for future usage
        log::debug!("name: {:?}", name);
    }
    log::info!("finished!");
    eprintln!("finished!");
    eprintln!("increase build?");
    eprintln!("ka ma lsp mode");
    let a: i32 = 212;
    println!("{:?}", a);

    Ok(())
}

async fn delete_dup() -> Result<()> {
    let graph = init_graph().await;
    for nl in NodeLabel::iter() {
        // with n.name is our unique key
        let cypher = format!(
            "MATCH (n:{}) \
                                WITH n.name as name, collect(n) AS nodes \
                                WHERE size(nodes) >  1 \
                                FOREACH (g in tail(nodes) | DETACH DELETE g);",
            nl
        );
        log::debug!("{:?}", cypher);

        let mut result = graph
            .execute(query(cypher.as_ref()))
            .await
            .map_err(|_| anyhow!("error on execute cypher"))?;

        // call next does real execute cypher
        while let Ok(Some(row)) = result.next().await {
            log::debug!("row is: {:#?}\n\n", row);
        }
    }
    Ok(())
}

async fn query_dup() -> Result<()> {
    let graph = init_graph().await;
    for nl in NodeLabel::iter() {
        let cypher = format!(
            "MATCH (n:{}) \
                            WITH n.name as name, collect(n) AS nodes \
                            WHERE size(nodes) >  1 \
                            RETURN name",
            nl
        );
        log::debug!("{:?}", cypher);

        let mut result = graph
            .execute(query(cypher.as_ref()))
            .await
            .map_err(|_| anyhow!("error on execute cypher"))?;

        while let Ok(Some(row)) = result.next().await {
            // log::debug!("row is: {:#?}\n\n", row);

            let name: String = row
                .get("name")
                .ok_or_else(|| anyhow!("no key `name` in row"))?;
            log::debug!("name: {:?}", name);
        }
        // break;
    }
    Ok(())
}

async fn quick_query() -> Result<()> {
    let graph = init_graph().await;
    let cypher = "
                MATCH (n:Symptom)
                WITH collect(n) AS nodes
                WHERE size(nodes) >  1
                WITH head(nodes) as n
                return n;
                ";
    log::debug!("{:?}", cypher);

    let mut result = graph
        .execute(query(cypher.as_ref()))
        .await
        .map_err(|_| anyhow!("error on execute cypher"))?;

    while let Ok(Some(row)) = result.next().await {
        log::debug!("row is: {:#?}\n\n", row);

        let node: Node = row.get("n").ok_or_else(|| anyhow!("no key `n` in row"))?;
        log::debug!("name: {:#?}", node);
        break;
    }
    Ok(())
}

async fn graph2file<P: AsRef<std::path::Path>>(file: P) -> Result<()> {
    use std::io::prelude::*;
    use std::io::BufWriter;

    let mut file = file.as_ref().to_owned();
    let mut opt = OpenOptions::new();
    if !file.is_absolute() {
        let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        file = workspace_dir.join(file);
    };

    dbg!(&file);

    // or use and_then
    if let Some(prefix) = file.parent() {
        std::fs::create_dir_all(prefix)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file)?;
    let mut stream = BufWriter::new(file);

    let graph = init_graph().await;
    // let cypher = "match (x) -[r]-> (y) return id(x) as id_x, id(y) as id_y, type(r)  as r;";
    let cypher = "match (x) -[r]-> (y) return head(labels(x)) as x_label, x.name as x, head(labels(y)) as y_label, y.name as y, type(r)  as r;";
    log::debug!("{}", cypher);

    let mut result = graph
        .execute(query(cypher.as_ref()))
        .await
        .map_err(|_| anyhow!("error on execute cypher"))?;

    let mut names_set = BTreeSet::new();
    let mut triples = Vec::new();
    while let Ok(Some(row)) = result.next().await {
        // log::debug!("row is: {:#?}\n\n", row);
        let x = row
            .get::<String>("x")
            .ok_or_else(|| anyhow!("no key `x` in row"))?;

        let y = row
            .get::<String>("y")
            .ok_or_else(|| anyhow!("no key `y` in row"))?;

        let x_label: String = row.get("x_label").ok_or_else(|| anyhow!("no key `x_label` in row"))?;
        let y_label: String = row.get("y_label").ok_or_else(|| anyhow!("no key `y_label` in row"))?;

        let r: String = row.get("r").ok_or_else(|| anyhow!("no key `r` in row"))?;

        let x = format!("{}::{}",x_label, x);
        let y = format!("{}::{}",y_label, y);
        names_set.insert(x.clone());
        names_set.insert(y.clone());
        triples.push((x, y, r));
    }
    // generate map & dump
    let id_name_map: HashMap<usize, String> = names_set.into_iter().enumerate().collect();
    let name_id_map: HashMap<String, usize> = id_name_map.iter().map(|(k,v)| (v.clone(), k.clone())).collect();

    // dump the map
    let map_string = serde_json::to_string(&id_name_map)?;
    std::fs::write("id_name_map.json", map_string)?;

    let map_string = serde_json::to_string(&name_id_map)?;
    std::fs::write("name_id_map.json", map_string)?;


    // write graph.txt
    for (x, y, r) in triples {
        let id_x = name_id_map.get(&x).unwrap();
        let id_y = name_id_map.get(&y).unwrap();
        writeln!(stream, "{} {} {}", id_x, id_y, r)?;
    }
    stream.flush()?;
    log::info!("finished!");
    Ok(())
}
