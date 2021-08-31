use actix_cors::Cors;
use actix_files::{self as fs, Files};
// use actix_web::{
//     error, get, guard, middleware, web, App, Error, HttpRequest, HttpResponse, HttpServer,
// };
use actix_web::{
    error::ErrorInternalServerError, error::InternalError, get, http::header, http::StatusCode,
    middleware, web, App, Error, HttpRequest, HttpResponse, HttpServer, Responder, Result,
};
// use anyhow::Result;
use platform::echarts;
use platform::kg::{self, Kg};
use platform::kg_d3::{self, Kg as Kg_d3};
use serde::Deserialize;
use std::{env, io};

/// greet and debug
async fn index(req: HttpRequest) -> &'static str {
    log::debug!("current dir: {:?}", std::env::current_dir());
    log::debug!("current file: {:?}", std::file!());
    log::debug!("root? {:?}", std::env!("CARGO_MANIFEST_DIR"));

    println!("REQ: {:?}", req);
    "Hello world!"
}

#[derive(Deserialize, Debug)]
struct NodeInfo {
    src_type: String,
    name: String,
}

#[get("/get_out_links")]
async fn get_out_links(web::Query(node_info): web::Query<NodeInfo>) -> Result<HttpResponse, Error> {
    println!("{:?}", node_info);
    let NodeInfo { src_type, name } = node_info;
    let res = Kg::convert(
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

#[get("/get_out_links_d3")]
async fn get_out_links_d3(
    web::Query(node_info): web::Query<NodeInfo>,
) -> Result<HttpResponse, Error> {
    println!("{:?}", node_info);
    let NodeInfo { src_type, name } = node_info;
    let res = Kg_d3::convert(
        src_type.as_str(),
        name.as_str(),
        dbg!(Kg_d3::get_out_links(src_type.as_str(), name.as_str()).await),
    )
    .map(|x| HttpResponse::Ok().json(x))
    .map_err(|e| {
        // InternalError::from_response("error", HttpResponse::InternalServerError().finish())
        ErrorInternalServerError(e)
    })?;
    Ok(res)
}

#[get("/get_stats")]
async fn get_stats() -> HttpResponse {
    let res = Kg::convert_stat(Kg::stat().await).await;
    HttpResponse::Ok().json(res)
}

#[get("/demo")]
async fn demo() -> Result<fs::NamedFile> {
    eprintln!("RUST_LOG : {:?}", std::env::var("RUST_LOG"));
    log::debug!("in demo: current dir: {:?}", std::env::current_dir());
    Ok(fs::NamedFile::open("static/html/demo.html")?.set_status_code(StatusCode::OK))
    // Ok(fs::NamedFile::open("static/html/demo.html")?)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // std::env::set_var("RUST_LOG", "actix_web=info");
    // let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    // std::env::set_current_dir(workspace_dir)?;
    env_logger::init();

    HttpServer::new(|| {
        App::new()
            // enable logger
            .wrap(
                Cors::default()
                    .allowed_origin("http://localhost:9091")
                    .allowed_origin("http://127.0.0.1:9091")
                    .allowed_methods(vec!["GET", "POST"])
                    .allowed_headers(vec![header::AUTHORIZATION, header::ACCEPT])
                    .allowed_header(header::CONTENT_TYPE)
                    .supports_credentials()
                    .max_age(3600),
            )
            .wrap(middleware::Logger::default())
            .service(Files::new("/static", "static").show_files_listing())
            .service(demo)
            .service(get_out_links)
            .service(get_out_links_d3)
            .service(get_stats)
            .service(web::resource("/index.html").to(|| async { "Hello world!" }))
            .service(web::resource("/").to(index))
    })
    .bind("0.0.0.0:9090")?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::dev::Service;
    use actix_web::{http, test, web, App, Error};

    #[actix_rt::test]
    async fn test_index() -> Result<(), Error> {
        let app = App::new().route("/", web::get().to(index));
        let mut app = test::init_service(app).await;

        let req = test::TestRequest::get().uri("/").to_request();
        let resp = app.call(req).await.unwrap();

        assert_eq!(resp.status(), http::StatusCode::OK);

        let response_body = match resp.response().body().as_ref() {
            Some(actix_web::body::Body::Bytes(bytes)) => bytes,
            _ => panic!("Response error"),
        };

        assert_eq!(response_body, r##"Hello world!"##);

        Ok(())
    }
}
