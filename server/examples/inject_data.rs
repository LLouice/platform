#![allow(dead_code)]

use neo4rs::*;
use platform::{neo4j::init_graph, triple::Triple};
// use std::sync::atomic::{AtomicU32, Ordering};
// use std::sync::Arc;
// use uuid::Uuid;
use futures::stream::{FuturesUnordered, StreamExt};
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::sync::Arc;
use tokio;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() {
    // inject_data().await.unwrap();
    // inject_sym().await.unwrap();
    inject3().await.unwrap();
    // inject_dis().await.unwrap();
    // inject_check().await.unwrap();
}

/// inject Symptom -> Area
async fn inject_data() -> Result<()> {
    let graph = init_graph().await;
    assert!(graph.run(query("RETURN 1")).await.is_ok());

    // delete all
    let txn = graph.start_txn().await?;
    let _ = txn
        .run_queries(vec![
            query("match (n:Symptom) detach delete n"),
            query("match (n:Area) detach delete n"),
        ])
        .await?;
    txn.commit().await?;

    // Constriant
    let txn = graph.start_txn().await?;
    txn.run_queries(vec![
        query("CREATE CONSTRAINT IF NOT EXISTS ON (s:Symptom) ASSERT s.name IS UNIQUE;"),
        query("CREATE CONSTRAINT IF NOT EXISTS ON (a:Area) ASSERT a.name IS UNIQUE;"),
    ])
    .await?;
    txn.commit().await?;

    let txn = graph.start_txn().await?;
    let filename = "data/chinese_symptom/all_result/split/症状相关部位.txt";
    for line in BufReader::new(File::open(filename)?).lines() {
        if let Ok(line) = line {
            let triple: Triple = serde_json::from_str(&line).expect("deserialize Triple error");
            txn.run_queries(vec![
                // Symptom
                query("MERGE(s:Symptom {name: $name}) SET s.updated_time=localdatetime()")
                    .param("name", triple.head.as_str()),
                // Area
                query("MERGE(a:Area{name: $name}) SET a.updated_time=localdatetime() RETURN a")
                    .param("name", triple.tail.as_str()),
                // SYMPTOM_RELATE_AREA
                query(
                    "MATCH(s:Symptom), (a:Area) WHERE s.name=$sname AND a.name=$aname
                    MERGE(s) -[r:SYMPTOM_RELATE_AREA]->(a)
                    SET r.updated_time=localdatetime()
                    RETURN r",
                )
                .param("sname", triple.head.as_str())
                .param("aname", triple.tail.as_str()),
            ])
            .await?;
        }
    }
    txn.commit().await?;
    Ok(())
}

/// use txn.run_queries
async fn inject_data2(head: &str, tail: &str, filename: &str, rel_name: &str) -> Result<()> {
    let graph = init_graph().await;
    assert!(graph.run(query("RETURN 1")).await.is_ok());

    // // delete all
    // let txn = graph.start_txn().await?;
    // let _ = txn
    //     .run_queries(vec![
    //         query("match (n:$name) delete n").param("name", head),
    //         query("match (n:$name) delete n").param("name", tail),
    //     ])
    //     .await?;
    // txn.commit().await?;

    // Constriant
    let txn = graph.start_txn().await?;
    let cypher_head = format!(
        "CREATE CONSTRAINT IF NOT EXISTS ON (s:{}) ASSERT s.name IS UNIQUE;",
        head
    );
    let cypher_tail = format!(
        "CREATE CONSTRAINT IF NOT EXISTS ON (s:{}) ASSERT s.name IS UNIQUE;",
        tail
    );
    txn.run_queries(vec![
        query(cypher_head.as_str()),
        query(cypher_tail.as_str()).param("name", tail),
    ])
    .await?;
    txn.commit().await?;

    let txn = graph.start_txn().await?;
    let txn = Arc::new(txn);
    let filename = format!("data/chinese_symptom/all_result/split/{}.txt", filename);
    dbg!(&filename);
    let now = std::time::Instant::now();
    let mut tasks = FuturesUnordered::new();
    for line in BufReader::new(File::open(filename)?).lines() {
        if let Ok(line) = line {
            let triple: Triple = serde_json::from_str(&line).expect("deserialize Triple error");

            // dbg!(&triple);
            let cypher_head = format!(
                "MERGE(s:{}{{name: $name}}) SET s.updated_time=localdatetime()",
                head,
            );
            // dbg!(&cypher_head);
            let cypher_tail = format!(
                "MERGE(s:{}{{name: $name}}) SET s.updated_time=localdatetime()",
                tail,
            );
            let cypher_rel = format!(
                "MATCH(s:{}), (a:{}) WHERE s.name=$sname AND a.name=$aname
                       MERGE(s) -[r:{}]->(a)
                       SET r.updated_time=localdatetime()
                       RETURN r",
                head, tail, rel_name
            );
            // txn.run_queries(vec![
            //     // Symptom
            //     query(cypher_head.as_str()).param("name", triple.head.as_str()),
            //     // Area
            //     query(cypher_tail.as_str()).param("name", triple.tail.as_str()),
            //     // SYMPTOM_RELATE_AREA
            //     query(cypher_rel.as_str())
            //         .param("sname", triple.head.as_str())
            //         .param("aname", triple.tail.as_str()),
            // ])
            // .await?;
            let txn_c = Arc::clone(&txn);
            let task = tokio::spawn(async move {
                txn_c
                    .run_queries(vec![
                        // Symptom
                        query(cypher_head.as_str()).param("name", triple.head.as_str()),
                        // Area
                        query(cypher_tail.as_str()).param("name", triple.tail.as_str()),
                        // SYMPTOM_RELATE_AREA
                        query(cypher_rel.as_str())
                            .param("sname", triple.head.as_str())
                            .param("aname", triple.tail.as_str()),
                    ])
                    .await
                    .unwrap()
            });
            tasks.push(task);
        }
    }
    while let Some(item) = tasks.next().await {
        let _ = item.unwrap();
    }
    // Arc::try_unwrap(txn).unwrap().commit().await?;
    if let Ok(txn) = Arc::try_unwrap(txn) {
        txn.commit().await?
    }
    println!("cost: {}", now.elapsed().as_secs());
    Ok(())
}

/// use graph.run
async fn inject_data3(
    head: &str,
    tail: &str,
    filename: &str,
    rel_name: &str,
    lock: Arc<Mutex<()>>,
) -> Result<()> {
    let graph = init_graph().await;
    assert!(graph.run(query("RETURN 1")).await.is_ok());

    // // delete all
    // let txn = graph.start_txn().await?;
    // let _ = txn
    //     .run_queries(vec![
    //         query("match (n:$name) delete n").param("name", head),
    //         query("match (n:$name) delete n").param("name", tail),
    //     ])
    //     .await?;
    // txn.commit().await?;

    // Constriant
    /*
    {
        let _lock = lock.lock().await;
        let txn = graph.start_txn().await?;
        let cypher_head = format!(
            "CREATE CONSTRAINT constraint_{}_name IF NOT EXISTS ON (s:{}) ASSERT s.name IS UNIQUE;",
            head,
            head
        );
        let cypher_tail = format!(
            "CREATE CONSTRAINT constraint_{}_name IF NOT EXISTS ON (s:{}) ASSERT s.name IS UNIQUE;",
            tail,
            tail
        );
        txn.run_queries(vec![
            query(cypher_head.as_str()),
            query(cypher_tail.as_str()).param("name", tail),
        ])
        .await;
        txn.commit().await;
    }
    */

    let graph = Arc::new(graph);
    let filename = format!("data/chinese_symptom/all_result/split/{}.txt", filename);
    dbg!(&filename);
    let now = std::time::Instant::now();
    let mut tasks = FuturesUnordered::new();
    for line in BufReader::new(File::open(filename)?).lines() {
        if let Ok(line) = line {
            let triple: Triple = serde_json::from_str(&line).expect("deserialize Triple error");

            // dbg!(&triple);
            let cypher_head = format!(
                "MERGE(s:{}{{name: $name}}) SET s.updated_time=localdatetime()",
                head,
            );
            // dbg!(&cypher_head);
            let cypher_tail = format!(
                "MERGE(s:{}{{name: $name}}) SET s.updated_time=localdatetime()",
                tail,
            );
            let cypher_rel = format!(
                "MATCH(s:{}), (a:{}) WHERE s.name=$sname AND a.name=$aname
                       MERGE(s) -[r:{}]->(a)
                       SET r.updated_time=localdatetime()
                       RETURN r",
                head, tail, rel_name
            );
            let graph_c = Arc::clone(&graph);

            let task = tokio::spawn(async move {
                let head_fut =
                    graph_c.run(query(cypher_head.as_str()).param("name", triple.head.as_str()));

                let tail_fut =
                    graph_c.run(query(cypher_tail.as_str()).param("name", triple.tail.as_str()));

                let (head_res, tail_res) = tokio::join!(head_fut, tail_fut);
                let _ = head_res.unwrap();
                let _ = tail_res.unwrap();

                graph_c
                    .run(
                        query(cypher_rel.as_str())
                            .param("sname", triple.head.as_str())
                            .param("aname", triple.tail.as_str()),
                    )
                    .await
                    .unwrap();
            });
            tasks.push(task);
        }
    }
    while let Some(item) = tasks.next().await {
        let _ = item.unwrap();
    }
    println!("cost: {}", now.elapsed().as_secs());
    Ok(())
}

async fn inject_sym() -> Result<()> {
    inject_data2(
        "Symptom",
        "Symptom",
        "症状相关症状",
        "SYMPTOM_RELATE_STMPTOM",
    )
    .await?;
    /*
    inject_data2("Symptom", "Area", "症状相关部位", "SYMPTOM_RELATE_AREA").await?;
    inject_data2("Symptom", "Drug", "症状相关药品", "SYMPTOM_RELATE_DRUG").await?;
    inject_data2("Symptom", "Check", "症状相关检查", "SYMPTOM_RELATE_CHECK").await?;
    inject_data2(
        "Symptom",
        "Disease",
        "症状相关疾病",
        "SYMPTOM_RELATE_DISEASE",
    )
    .await?;
    inject_data2(
        "Symptom",
        "Department",
        "症状相关科室",
        "SYMPTOM_RELATE_DEPARTMENT",
    )
    .await?;
    */
    Ok(())
}

async fn inject_dis() -> Result<()> {
    /*
    inject_data2("Disease", "Area", "疾病相关部位", "DISEASE_RELATE_AREA").await?;
    inject_data2("Disease", "Drug", "疾病相关药品", "DISEASE_RELATE_DRUG").await?;
    inject_data2("Disease", "Check", "疾病相关检查", "DISEASE_RELATE_CHECK").await?;
    inject_data2(
        "Disease",
        "Disease",
        "疾病相关疾病",
        "DISEASE_RELATE_DISEASE",
    )
    .await?;
    inject_data2(
        "Disease",
        "Department",
        "疾病相关科室",
        "DISEASE_RELATE_DEPARTMENT",
    )
    .await?;
    */
    inject_data2(
        "Disease",
        "Symptom",
        "疾病相关症状",
        "DISEASE_RELATE_SYMPTOM",
    )
    .await?;
    Ok(())
}

async fn inject_check() -> Result<()> {
    inject_data2("Check", "Area", "检查相关部位", "CHECK_RELATE_AREA").await?;
    inject_data2("Check", "Check", "检查相关检查", "CHECK_RELATE_CHECK").await?;
    inject_data2("Check", "Disease", "检查相关疾病", "CHECK_RELATE_DISEASE").await?;
    inject_data2(
        "Check",
        "Department",
        "检查相关科室",
        "CHECK_RELATE_DEPARTMENT",
    )
    .await?;
    inject_data2("Check", "Symptom", "检查相关症状", "CHECK_RELATE_SYMPTOM").await?;
    Ok(())
}

async fn create_constraints() -> Result<()> {
    let graph = init_graph().await;
    let txn = graph.start_txn().await?;
    let cyphers = [
        "CREATE CONSTRAINT constraint_Symptom_name IF NOT EXISTS ON (s:Symptom) ASSERT s.name IS UNIQUE;",
        "CREATE CONSTRAINT constraint_Disease_name IF NOT EXISTS ON (s:Disease) ASSERT s.name IS UNIQUE;",
        "CREATE CONSTRAINT constraint_Drug_name IF NOT EXISTS ON (s:Drug) ASSERT s.name IS UNIQUE;",
        "CREATE CONSTRAINT constraint_Department_name IF NOT EXISTS ON (s:Department) ASSERT s.name IS UNIQUE;",
         "CREATE CONSTRAINT constraint_Check_name IF NOT EXISTS ON (s:Check) ASSERT s.name IS UNIQUE;",
        "CREATE CONSTRAINT constraint_Area_name IF NOT EXISTS ON (a:Area) ASSERT a.name IS UNIQUE;"
    ];
    txn.run_queries(cyphers.map(query).to_vec()).await;
    txn.commit().await;

    Ok(())
}

/// inject all Symptom relations but Symptom itself
async fn inject3() -> Result<()> {
    // create all constraints first;
    create_constraints()
        .await
        .map_err(|e| {eprintln!("{:#?}", e); e})
        .expect("failed to create constraints");

    let mut tasks = FuturesUnordered::new();

    let lock = Arc::new(Mutex::new(()));

    tasks.push(inject_data3(
        "Symptom",
        "Area",
        "症状相关部位",
        "SYMPTOM_RELATE_AREA",
        Arc::clone(&lock),
    ));
    tasks.push(inject_data3(
        "Symptom",
        "Drug",
        "症状相关药品",
        "SYMPTOM_RELATE_DRUG",
        Arc::clone(&lock),
    ));
    tasks.push(inject_data3(
        "Symptom",
        "Check",
        "症状相关检查",
        "SYMPTOM_RELATE_CHECK",
        Arc::clone(&lock),
    ));
    tasks.push(inject_data3(
        "Symptom",
        "Disease",
        "症状相关疾病",
        "SYMPTOM_RELATE_DISEASE",
        Arc::clone(&lock),
    ));
    tasks.push(inject_data3(
        "Symptom",
        "Department",
        "症状相关科室",
        "SYMPTOM_RELATE_DEPARTMENT",
        Arc::clone(&lock),
    ));

    while let Some(item) = tasks.next().await {
        let _ = item.unwrap();
    }
    Ok(())
}
