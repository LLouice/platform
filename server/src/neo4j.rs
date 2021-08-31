use neo4rs::*;

pub async fn init_graph() -> Graph {
    // let host = "10.108.211.136";
    let host = "127.0.0.1";
    let port = "7687";
    let uri = format!("{}:{}", host, port);
    let user = "neo4j";
    let pass = "symptom";
    Graph::new(&uri, user, pass).await.unwrap()
}
