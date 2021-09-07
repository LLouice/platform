use wasm_bindgen_futures::{spawn_local, JsFuture};
use wasm_logger;

use wasm_bindgen::JsCast;

use reqwasm::http::{Request, Response};
use web_sys::HtmlInputElement;

use yew::html::Scope;
use yew::prelude::*;

use anyhow::Result;

use platform::GraphData;

use crate::bindings;

struct App;

pub(crate) enum AppMsg {
    Search(String),
}

impl Component for App {
    type Message = AppMsg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
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

    fn changed(&mut self, ctx: &Context<Self>) -> bool {
        false
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <>
                { self.view_navbar(ctx.link()) }

                <div class="block">
                    <div class="container network">
                        <div id="network" style="width:1200px;height:1000px;margin:auto"></div>
                        <div id="stats_rels" style="width:1200px;height:1000px;margin:auto"></div>
                    </div>
                </div>
            </>
        }
    }
}

impl App {
    fn view_navbar(&self, link: &Scope<Self>) -> Html {
        let onkeypress = link.batch_callback(|e: KeyboardEvent| {
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
        // let resp = spawn_local(async move {
        //     let uri = format!(
        //         "http://localhost:9090/get_out_links?src_type=Symptom&name={}",
        //         name
        //     );

        //     let resp = Request::get(&uri).send().await.unwrap();
        //     log::info!("{:?}", resp);
        //     let graph_data: GraphData = resp.json().await.unwrap();
        //     log::info!("graph_data: {:?}", graph_data);
        // });

        unsafe {
            spawn_local(bindings::display_network("Symptom".to_string(), name));
        }
    }
}

pub fn run() {
    // yew::initialize();
    wasm_logger::init(wasm_logger::Config::default());
    // env_logger::init();
    // spawn_local(bindings::display_network());

    log::info!("start yew app");
    let mount_point = yew::utils::document()
        .query_selector("#yew")
        .unwrap()
        .unwrap();

    unsafe {
        bindings::main();
    }

    // only required for stdweb
    // yew::run_loop();

    yew::start_app_in_element::<App>(mount_point);
    log::info!("after start app");
}
