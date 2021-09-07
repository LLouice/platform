use wasm_bindgen_futures::{spawn_local, JsFuture};
use wasm_logger;

use wasm_bindgen::JsCast;

use web_sys::HtmlInputElement;

use yew::format::{Json, Nothing};
use yew::prelude::*;
use yew::services::fetch::{FetchService, FetchTask, Request, Response};

use ybc::NavbarFixed::Top;
use ybc::TileCtx::{Ancestor, Child, Parent};
use ybc::TileSize::Four;

use anyhow::Result;

use platform::GraphData;

use crate::bindings;

struct App {
    link: ComponentLink<Self>,
    fetch_task: Option<FetchTask>,
}

pub(crate) enum AppMsg {
    Search(String),
    ReceiveGraphData(Result<GraphData>),
}

impl Component for App {
    type Message = AppMsg;
    type Properties = ();

    fn create(_props: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
            link,
            fetch_task: None,
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        use AppMsg::*;
        match msg {
            Search(value) => {
                log::info!("enter input the input is {:?}", value);
                self.search(value);
            }
            _ => {}
        }

        false
    }

    fn change(&mut self, _props: Self::Properties) -> ShouldRender {
        false
    }

    fn view(&self) -> Html {
        html! {
            <>
                { self.view_navbar() }

                <ybc::Block>
                    <ybc::Container classes={classes!("network")}>
                        <div id="network" style="width:1200px;height:1000px;margin:auto"></div>

                        <div id="stats_rels" style="width:1200px;height:1000px;margin:auto"></div>
                    </ybc::Container>

                </ybc::Block>


            </>
        }
    }
}

impl App {
    fn view_navbar(&self) -> Html {
        let onkeypress = self.link.batch_callback(|e: KeyboardEvent| {
            if e.key() == "Enter" {
                let input = e
                    .target()
                    .and_then(|t| t.dyn_into::<HtmlInputElement>().ok());
                let value = input.map(|x| x.value());
                // input.set_value("");

                value.map(AppMsg::Search)
            } else {
                None
            }
        });

        html! {
            <section class="navbar">
                <nav class="navbar" role="navigation" aria-label="main navigation">
                    <div class="navbar-brand">
                        <a class="navbar-item" href="https://bulma.io">
                            <img src="https://bulma.io/images/bulma-logo.png" width="112" height="28"/>
                        </a>

                        <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false" data-target="navbarBasicExample">
                            <span aria-hidden="true"></span>
                            <span aria-hidden="true"></span>
                            <span aria-hidden="true"></span>
                        </a>
                    </div>

                    <div id="navbarBasicExample" class="navbar-menu">
                        <div class="navbar-start">
                            <a class="navbar-item">
                                { "Home" }
                            </a>

                            <a class="navbar-item">
                                { "Documentation" }
                            </a>

                            <div class="navbar-item has-dropdown is-hoverable">
                                <a class="navbar-link">
                                    { "More" }
                                </a>

                                <div class="navbar-dropdown">
                                    <a class="navbar-item">
                                        { "About" }
                                    </a>
                                    <a class="navbar-item">
                                        { "Jobs" }
                                    </a>
                                    <a class="navbar-item">
                                        { "Contact" }
                                    </a>
                                    <hr class="navbar-divider"/>
                                    <a class="navbar-item">
                                        { "Report an issue" }
                                    </a>
                                </div>
                            </div>
                        </div>

                        <div class="navbar-end">
                            <div class="navbar-item">
                                <div class="field">
                                    <p class="control has-icons-right">
                <input class="input is-success" type="text" placeholder="肩背痛"  onkeypress={onkeypress} />
                                        <span class="icon is-small is-right">
                                            <i class="fas fa-search"></i>
                                        </span>
                                    </p>

                                </div>
                            </div>
                        </div>
                    </div>
                </nav>
            </section>
        }
    }
}

impl App {
    fn search(&self, name: String) {
        let uri = format!(
            "http://localhost:9090/get_out_links_d3?src_type=Symptom&name={}",
            name
        );
        let get_request = Request::get(uri)
            .body(Nothing)
            .expect("Could not build that request");
        let callback = self
            .link
            .callback(|response: Response<Json<Result<GraphData>>>| {
                let Json(data) = response.into_body();
                AppMsg::ReceiveGraphData(data)
            });
        let task = FetchService::fetch(get_request, callback).expect("failed to start request");
        // 4. store the task so it isn't canceled immediately
        self.fetch_task = Some(task);
    }
}

pub fn run() {
    yew::initialize();
    wasm_logger::init(wasm_logger::Config::default());
    // env_logger::init();
    // spawn_local(bindings::display_network());

    log::info!("start yew app");
    let mount_point = yew::utils::document()
        .query_selector("#yew")
        .unwrap()
        .unwrap();
    yew::App::<App>::new().mount(mount_point);

    unsafe {
        bindings::main();
    }

    // only required for stdweb
    yew::run_loop();
}
