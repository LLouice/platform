//! insert data to Nebula Graph.
#[macro_use]
extern crate anyhow;
use anyhow::Result;
use fbthrift_transport::{tokio_io::transport::AsyncTransport, AsyncTransportConfiguration};
use nebula_client::v2::{GraphClient, GraphTransportResponseHandler};
use platform::resp::Resp;
use platform::utils::get_current_timestamp;
use std::env;
use tokio::net::TcpStream;

macro_rules! exec {
    ($session:expr, $cmd:expr) => {{
        let resp = $session.execute(&$cmd.as_bytes().to_vec()).await?;
        println!("{:?}", resp);
        Resp::parse(resp)
    }};
}

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

    let session = client
        .authenticate(&username.as_bytes().to_vec(), &password.as_bytes().to_vec())
        .await;

    if let Ok(mut session) = session {
        session.execute(&b"USE nba".to_vec()).await?;

        // symptom
        // drop first
        // let resp = session.execute(&b"DROP TAG symptom".to_vec()).await?;
        // let res = Resp::parse(resp);
        // println!("{:?}", res);

        // let cmd = r#"CREATE TAG IF NOT EXISTS symptom(name string NOT NULL, created_time timestamp NOT NULL)"#;
        // let res = exec!(session, cmd);
        // println!("{:?}", res);

        // let cmd = r#"CREATE EDGE IF NOT EXISTS sym_relate_area(created_time timestamp NOT NULL)"#;
        // let res = exec!(session, cmd);
        // println!("{:?}", res);

        // let cmd = r#"CREATE TAG IF NOT EXISTS area(name string NOT NULL, created_time timestamp NOT NULL)"#;
        // let res = exec!(session, cmd);
        // println!("{:?}", res);

        // let cmd = r#"INSERT VERTEX symptom(name, created_time) VALUE "1000":("test_sym", timestamp("2000-01-01T08:00:00"));"#;
        // println!("{}", cmd);
        // // println!("{:?}", cmd);
        // let res = exec!(session, cmd);
        // println!("{:?}", res);

        // let cmd = r#"FETCH PROP on player "100""#;
        // // println!("{:?}", cmd);
        // let res = exec!(session, cmd);
        // println!("{:?}", res);

        let cmd = r#"FETCH PROP on symptom "1000""#;
        println!("{}", cmd);
        // println!("{:?}", cmd);
        let res = exec!(session, cmd);
        println!("{:?}", res);

        println!("done");
        Ok(())
    } else {
        println!("connect error!");
        bail!("connect error!");
    }
}
