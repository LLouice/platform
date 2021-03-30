use actix_files::{self as fs, Files};
// use actix_web::{
//     error, get, guard, middleware, web, App, Error, HttpRequest, HttpResponse, HttpServer,
// };
use actix_web::{
    get, http::StatusCode, middleware, web, App, Error, HttpRequest, HttpResponse, HttpServer,
    Responder, Result,
};
// use anyhow::Result;
use platform::echarts;
use platform::kg::{self, Kg};
use serde::Deserialize;
use std::{env, io};

async fn index(req: HttpRequest) -> &'static str {
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
    .map_err(|_| HttpResponse::InternalServerError())?;
    Ok(res)
}

#[get("/get_stats")]
async fn get_stats() -> HttpResponse {
    let res = Kg::convert_stat(Kg::stat().await).await;
    HttpResponse::Ok().json(res)
}

#[get("/demo")]
async fn demo() -> Result<fs::NamedFile> {
    Ok(fs::NamedFile::open("static/html/demo.html")?.set_status_code(StatusCode::OK))
    // Ok(fs::NamedFile::open("static/html/demo.html")?)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    HttpServer::new(|| {
        App::new()
            // enable logger
            .wrap(middleware::Logger::default())
            .service(Files::new("/static", "static").show_files_listing())
            .service(demo)
            .service(get_out_links)
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
