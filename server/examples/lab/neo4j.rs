use futures::stream::FuturesUnordered;
use futures::stream::*;
use neo4rs::*;
use platform::neo4j::init_graph;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use uuid::Uuid;

#[tokio::main]
async fn main() {
    // run_basic();
    run_create_node();
}

async fn run_basic() {
    let graph = init_graph().await;
    let graph = Arc::new(graph);

    let id = Uuid::new_v4().to_string();
    let mut result = graph
        .run(query("CREATE (p:Person {id: $id})").param("id", id.clone()))
        .await
        .unwrap();

    let mut count = Arc::new(AtomicU32::new(0));
    let mut handles = FuturesUnordered::new();
    for _ in 1..=42 {
        let graph = graph.clone();
        let id = id.clone();
        let count = count.clone();
        let handle = tokio::spawn(async move {
            let mut result = graph
                .execute(query("MATCH (p:Person {id: $id}) RETURN p").param("id", id))
                .await
                .unwrap();
            while let Ok(Some(row)) = result.next().await {
                println!("row is {:?}", row);
                count.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    // futures::future::join_all(handles).await;
    while let Some(item) = handles.next().await {
        let () = item.unwrap();
    }

    assert_eq!(count.load(Ordering::Relaxed), 42);
}

async fn run_create_node() {
    let graph = init_graph().await;

    assert!(graph.run(query("RETURN 1")).await.is_ok());

    let mut result = graph
        .execute(
            query("CREATE (friend:Person {name: $name}) RETURN friend").param("name", "Mr Mark"),
        )
        .await
        .unwrap();

    while let Ok(Some(row)) = result.next().await {
        let node: Node = row.get("friend").unwrap();
        let id = node.id();
        let labels = node.labels();
        let name: String = node.get("name").unwrap();
        assert_eq!(name, "Mr Mark");
        assert_eq!(labels, vec!["Person"]);
        assert!(id > 0);
    }
}
