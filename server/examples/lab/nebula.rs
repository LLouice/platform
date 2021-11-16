/*
cargo run -p nebula-demo-tokio --bin v2_graph_client 127.0.0.1 9669 user 'password'
*/

#[macro_use]
extern crate anyhow;
use anyhow::Result;
use fbthrift_transport::{tokio_io::transport::AsyncTransport, AsyncTransportConfiguration};
use nebula_client::v2::{GraphClient, GraphTransportResponseHandler};
use platform::resp::Resp;
use std::env;
use tokio::net::TcpStream;

#[tokio::main]
async fn main() -> Result<()> {
    run().await
}

async fn run() -> Result<()> {
    let domain = env::args()
        .nth(1)
        .unwrap_or_else(|| env::var("DOMAIN").unwrap_or_else(|_| "10.108.211.136".to_owned()));
    let port: u16 = env::args()
        .nth(2)
        .unwrap_or_else(|| env::var("PORT").unwrap_or_else(|_| "9669".to_owned()))
        .parse()
        .unwrap();
    let username = env::args()
        .nth(3)
        .unwrap_or_else(|| env::var("USERNAME").unwrap_or_else(|_| "user".to_owned()));
    let password = env::args()
        .nth(4)
        .unwrap_or_else(|| env::var("PASSWORD").unwrap_or_else(|_| "password".to_owned()));

    println!(
        "v2_graph_client {} {} {} {}",
        domain, port, username, password
    );

    //
    let addr = format!("{}:{}", domain, port);
    let stream = TcpStream::connect(addr).await?;

    //
    let transport = AsyncTransport::new(
        stream,
        AsyncTransportConfiguration::new(GraphTransportResponseHandler),
    );
    let client = GraphClient::new(transport);

    // let mut session = client
    //     .authenticate(&username.as_bytes().to_vec(), &password.as_bytes().to_vec())
    //     .await?;
    let session = client
        .authenticate(&username.as_bytes().to_vec(), &password.as_bytes().to_vec())
        .await;

    if let Ok(mut session) = session {
        session.execute(&b"USE nba".to_vec()).await?;

        let resp = session.execute(&b"SHOW TAGS".to_vec()).await?;
        let res = Resp::parse(resp);
        // println!("connect it!");
        println!("{:?}", res);

        let resp = session.execute(&b"DESC TAG player;".to_vec()).await?;
        println!("{:?}", resp);
        let res = Resp::parse(resp);
        println!("{:?}", res);

        println!("------------------");
        let resp = session.execute(&b"DESC TAG player;".to_vec()).await?;
        let res = Resp::parse(resp);
        println!("{:?}", res);

        println!("done");
        Ok(())
    } else {
        println!("connect error!");
        bail!("connect error!");
    }
}
