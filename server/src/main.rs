#[macro_use]
extern crate platform;

use std::{env, io};
// use anyhow::Result; use serde::Deserialize;
use std::collections::HashSet;
use std::future::Future;

use actix_cors::Cors;
use actix_files::{self as fs, Files};
// use actix_web::{
//     error, get, guard, middleware, web, App, Error, HttpRequest, HttpResponse, HttpServer,
// };
use actix_redis::RedisSession;
use actix_session::{CookieSession, Session};
// use actix_sled_session::{Session, SledSession};
use actix_web::{
    App,
    error::{ErrorBadRequest, ErrorInternalServerError, InternalError},
    Error,
    get,
    http::header, http::StatusCode, HttpRequest, HttpResponse, HttpServer, middleware, post, Responder, Result, web,
};
use futures_util::stream::StreamExt as _;
use rand::Rng;
use serde::{Deserialize, Serialize};

use platform::kg::{self, IncreaseUpdateState, Kg, NodeInfo, QRandomSample};
use platform::session::GraphSession;

const GRAPHSESSION: &'static str = "GraphSession";

/// greet and debug
async fn index(req: HttpRequest) -> &'static str {
    log::debug!("current dir: {:?}", std::env::current_dir());
    log::debug!("current file: {:?}", std::file!());
    log::debug!("root? {:?}", std::env!("CARGO_MANIFEST_DIR"));

    println!("REQ: {:?}", req);
    "Hello world!"
}

#[get("/query_links")]
async fn query_links(
    web::Query(node_info): web::Query<NodeInfo>,
    session: Session,
) -> Result<HttpResponse, Error> {
    Kg::query_node_links(&node_info).await;

    let NodeInfo { src_type, name } = &node_info;

    let res = Kg::convert_dedup(
        src_type.as_str(),
        name.as_str(),
        Kg::query_node_links(&node_info).await,
    );

    // ugly unpack
    match res {
        Ok((graph_data, graph_sess)) => {
            log::debug!(
                "graph session: {:#?}\n\tnode_keys len: {}",
                graph_sess,
                graph_sess.node_keys.len()
            );
            let res = session.insert(GRAPHSESSION.to_string(), graph_sess);
            // the can't inser error? why?
            if let Err(e) = res {
                log::error!("session insert error: {:?}", e);
            }
            assert!(
                session.get::<GraphSession>(GRAPHSESSION)?.is_some(),
                "inserted but no data there"
            );
            Ok(HttpResponse::Ok().json(graph_data))
        }

        Err(e) => Err(ErrorInternalServerError(e)),
    }
}

#[get("/get_out_links")]
async fn get_out_links(
    web::Query(node_info): web::Query<NodeInfo>,
    session: Session,
) -> Result<HttpResponse, Error> {
    log::info!("in get_out_links");
    // TODO add src_type as key, for preventing from set data again
    log::debug!("{:?}", node_info);

    log::debug!("get session: {:?}", session.entries());
    if let Some(graph_sess) = session.get::<GraphSession>(GRAPHSESSION)? {
        log::debug!("pre graph session: {:#?}", graph_sess);
    } else {
        log::error!("no graph session");
    }

    let NodeInfo { src_type, name } = node_info;
    let res = Kg::convert_dedup(
        src_type.as_str(),
        name.as_str(),
        Kg::get_out_links(src_type.as_str(), name.as_str()).await,
    );

    // ugly unpack
    match res {
        Ok((graph_data, graph_sess)) => {
            log::debug!(
                "graph session: {:#?}\n\tnode_keys len: {}",
                graph_sess,
                graph_sess.node_keys.len()
            );
            let res = session.insert(GRAPHSESSION.to_string(), graph_sess);
            // the can't inser error? why?
            if let Err(e) = res {
                log::error!("session insert error: {:?}", e);
            }
            assert!(
                session.get::<GraphSession>(GRAPHSESSION)?.is_some(),
                "inserted but no data there"
            );
            Ok(HttpResponse::Ok().json(graph_data))
        }

        Err(e) => Err(ErrorInternalServerError(e)),
    }

    // let res = res.map(|x| {
    //     HttpResponse::Ok().json(x)
    // })
    // .map_err(|e| {
    //     // InternalError::from_response("error", HttpResponse::InternalServerError().finish())
    //     ErrorInternalServerError(e)
    // })?;
    // Ok(res)
}

#[get("/get_out_links_d3")]
async fn get_out_links_d3(
    web::Query(node_info): web::Query<NodeInfo>,
) -> Result<HttpResponse, Error> {
    println!("{:?}", node_info);
    let NodeInfo { src_type, name } = node_info;
    let res = Kg::convert_d3_dedup(
        src_type.as_str(),
        name.as_str(),
        dbg!(Kg::get_out_links(src_type.as_str(), name.as_str()).await),
    )
    .map(|x| HttpResponse::Ok().json(x))
    .map_err(|e| {
        // InternalError::from_response("error", HttpResponse::InternalServerError().finish())
        ErrorInternalServerError(e)
    })?;
    Ok(res)
}

// for debug
#[get("/debug/get_out_links")]
async fn _get_out_links(web::Query(node_info): web::Query<NodeInfo>) -> impl Responder {
    println!("{:?}", node_info);
    let NodeInfo { src_type, name } = node_info;
    // let res = Kg::convert_dedup(
    //     src_type.as_str(),
    //     name.as_str(),
    //     dbg!(Kg::get_out_links(src_type.as_str(), name.as_str()).await),
    // )
    let res = dbg!(Kg::get_out_links(src_type.as_str(), name.as_str()).await);
    web::Json(res)
}

#[get("/get_in_links")]
async fn get_in_links(web::Query(node_info): web::Query<NodeInfo>) -> impl Responder {
    println!("{:?}", node_info);
    let NodeInfo { src_type, name } = node_info;
    // let res = Kg::convert_dedup(
    //     src_type.as_str(),
    //     name.as_str(),
    //     dbg!(Kg::get_out_links(src_type.as_str(), name.as_str()).await),
    // )
    let res = dbg!(Kg::get_in_links(src_type.as_str(), name.as_str()).await);
    web::Json(res)
}

#[post("/increase_update")]
async fn increase_update(
    increase_update_state: web::Json<IncreaseUpdateState>,
    sess: Session,
) -> Result<HttpResponse, Error> {
    if let Some(graph_sess) = sess.get::<GraphSession>(GRAPHSESSION)? {
        log::debug!(
            "pre graph session: {:#?},\n\tnode_keys len: {}",
            graph_sess,
            graph_sess.node_keys.len()
        );

        let IncreaseUpdateState {
            node_info,
            current_nodes_id,
            current_links_pair,
        } = increase_update_state.into_inner();

        let kg_res = Kg::query_node_links(&node_info).await;

        let current_nodes_id: HashSet<usize> = current_nodes_id.into_iter().collect();
        let current_links_set: HashSet<(usize, usize)> = current_links_pair.into_iter().collect();

        let res = Kg::convert_dedup_with_session(
            &node_info,
            current_nodes_id,
            current_links_set,
            kg_res,
            graph_sess,
        );

        match res {
            Ok((graph_data, graph_sess_new)) => {
                log::debug!(
                    "graph session: {:#?}\n\tnode_keys len: {}",
                    graph_sess_new,
                    graph_sess_new.node_keys.len()
                );
                let res = sess.insert(GRAPHSESSION.to_string(), graph_sess_new);
                if let Err(e) = res {
                    log::error!("session insert error: {:?}", e);
                }
                Ok(HttpResponse::Ok().json(graph_data))
            }

            Err(e) => Err(ErrorInternalServerError(e)),
        }
    } else {
        // FIXME return InternalError
        log::error!("no graph session");
        Err(ErrorBadRequest("no grpah session"))
    }
}

#[get("/get_stats")]
async fn get_stats() -> HttpResponse {
    let res = Kg::convert_stat(Kg::stat().await).await;
    HttpResponse::Ok().json(res)
}


#[get("/random_sample")]
async fn random_sample(
    web::Query(query): web::Query<QRandomSample>) -> Result<HttpResponse, Error> {
    // get query info
    let res = Kg::random_sample(query).await
        .map(|x| HttpResponse::Ok().json(x))
        .map_err(|e| {
            // InternalError::from_response("error", HttpResponse::InternalServerError().finish())
            ErrorInternalServerError(e)
        })?;
    Ok(res)
    // Ok(HttpResponse::Ok().json(query))
}

#[get("/demo")]
async fn demo() -> Result<fs::NamedFile> {
    eprintln!("RUST_LOG : {:?}", std::env::var("RUST_LOG"));
    log::debug!("in demo: current dir: {:?}", std::env::current_dir());
    Ok(fs::NamedFile::open("static/html/demo.html")?.set_status_code(StatusCode::OK))
    // Ok(fs::NamedFile::open("static/html/demo.html")?)
}

#[get("/debug/session")]
async fn _session(session: Session) -> Result<&'static str, Error> {
    // access session data
    if let Some(count) = session.get::<i32>("counter")? {
        log::debug!("SESSION value: {}", count);
        session.insert("counter", count + 1)?;
    } else {
        session.insert("counter", 1)?;
    }
    log::debug!(
        "GraphSession: {:?}",
        session.get::<GraphSession>(GRAPHSESSION)?
    );

    Ok("Welcome!")
}

/// session is based on spectified url? maybe no
#[get("/debug/session2")]
async fn _session2(session: Session) -> Result<&'static str, Error> {
    // access session data
    //

    log::debug!("counter session: {:?}", session.get::<i32>("counter")?);

    log::debug!(
        "GraphSession: {:?}",
        session.get::<GraphSession>(GRAPHSESSION)?
    );

    Ok("Welcome!")
}

#[derive(Deserialize, Debug)]
struct Foo {
    x: usize,
}

#[post("/debug/post-json")]
async fn post_json(foo: web::Json<Foo>, sess: Session) -> impl Responder {
    let foo = format!("{:?}", foo);
    eprintln!("foo: {:?}", foo);
    foo
}

//// Handler can have up to 12 extractor, order doesn't matter!
/// inpect post payload
#[post("/debug/inspect-post")]
async fn _inspect_post(mut body: web::Payload) -> Result<String> {
    let mut bytes = web::BytesMut::new();
    while let Some(item) = body.next().await {
        bytes.extend_from_slice(&item?);
    }
    let info = serde_json::from_slice::<Foo>(&bytes);
    eprintln!("info: {:?}", info);
    Ok(format!("Request Body Bytes:\n{:?}", bytes))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // std::env::set_var("RUST_LOG", "actix_web=info");
    // let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    // std::env::set_current_dir(workspace_dir)?;

    platform::init_env_logger!();

    //     use std::io::Write;
    //     use log::LevelFilter;
    //
    //     env_logger::Builder::new()
    //             .format(|buf, record| {
    //                 writeln!(
    //                     buf,
    //                     "{}:{} {} [{}] - {}",
    //                     record.file().unwrap_or("unknown"),
    //                     record.line().unwrap_or(0),
    //                     chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
    //                     record.level(),
    //                     record.args()
    //                 )
    //             })
    //             .filter(None, LevelFilter::Debug)
    //             .init();

    eprintln!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));

    // Generate a random 32 byte key. Note that it is important to use a unique
    // private key for every project. Anyone with access to the key can generate
    // authentication cookies for any user!
    let private_key = rand::thread_rng().gen::<[u8; 32]>();
    // let session_backend = SledSession::new_default()?;

    HttpServer::new(move || {
        App::new()
            // enable logger
            .wrap(
                // Cors::default()
                Cors::permissive()
                    // .allowed_origin("*")
                    // .allowed_origin("http://localhost:9091")
                    // .allowed_origin("http://127.0.0.1:9091")
                    // .allowed_origin("http://localhost:9092")
                    // .allowed_origin("http://127.0.0.1:9092")
                    // .allowed_methods(vec!["GET", "POST"])
                    // .allowed_headers(vec![header::AUTHORIZATION, header::ACCEPT])
                    // .allowed_header(header::CONTENT_TYPE)
                    .supports_credentials()
                    .max_age(3600),
            )
            // redis session middleware
            // .wrap(session_backend.clone())
            .wrap(RedisSession::new("127.0.0.1:6379", &private_key))
            // .wrap(CookieSession::signed(&[0; 32]).secure(false))
            .wrap(middleware::Logger::default())
            .service(Files::new("/static", "static").show_files_listing())
            .service(demo)
            .service(query_links)
            .service(get_out_links)
            .service(get_out_links_d3)
            .service(increase_update)
            .service(get_stats)
            .service(random_sample)
            // for debug
            .service(_get_out_links)
            .service(get_in_links)
            .service(_session)
            .service(_session2)
            .service(post_json)
            .service(_inspect_post)
            // static file
            .service(web::resource("/index.html").to(|| async { "Hello world!" }))
            .service(web::resource("/").to(index))
    })
    .bind("0.0.0.0:9090")?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use actix_web::{App, Error, http, test, web};
    use actix_web::dev::Service;

    use super::*;

    #[actix_rt::test]
    async fn test_index() -> Result<(), Error> {
        let app = App::new().route("/", web::get().to(index));
        let mut app = test::init_service(app).await;

        let req = test::TestRequest::get().uri("/").to_request();
        let resp = app.call(req).await.unwrap();

        assert_eq!(resp.status(), http::StatusCode::OK);

        let response_body = match resp.response().body() {
            Some(actix_web::body::Body::Bytes(bytes)) => bytes,
            _ => panic!("Response error"),
        };

        assert_eq!(response_body, r##"Hello world!"##);

        Ok(())
    }
}
