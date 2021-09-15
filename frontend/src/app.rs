use anyhow::Result;
use reqwasm::http::{Request, Response};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{JsFuture, spawn_local};
use wasm_logger;
use web_sys::HtmlInputElement;
use yew::html::Scope;
use yew::prelude::*;
use yew_router::prelude::*;

use platform::GraphData;

use crate::bindings;
use crate::pages::{home::Home, page_not_found::PageNotFound, symptom::PageSymptom};

#[derive(Debug, Default)]
struct App {
    search_inp: NodeRef,
    search_cat: Category,
    search_placeholder: String,
    navbar_active: bool,
}

#[derive(Debug, Clone)]
pub enum Category {
    Symptom,
    Disease,
    Drug,
    Department,
    Check,
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cat = match self {
            Category::Symptom => "Symptom",
            Category::Disease => "Disease",
            Category::Drug => "Drug",
            Category::Department => "Department",
            Category::Check => "Check",
        };
        write!(f, "{}", cat)
    }
}

impl Default for Category {
    fn default() -> Self {
        Category::Symptom
    }
}

#[derive(Routable, PartialEq, Clone, Debug)]
pub enum Route {
    // #[at("/posts/:id")]
    // Post { id: u64 },
    #[at("/symptom")]
    Symptom,
    #[at("/")]
    Home,
    #[not_found]
    #[at("/404")]
    NotFound,
}

pub(crate) enum AppMsg {
    Nop,
    // toggle
    ToggleNavbar,
    Search(String),
    ChangeSearchCat(Category),
    FillPlaceholder,
}

impl Component for App {
    type Message = AppMsg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            search_cat: Category::Symptom,
            search_placeholder: String::from("肩背痛"),
            navbar_active: false,
            ..Default::default()
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        use AppMsg::*;

        match msg {
            // toggle
            ToggleNavbar => {
                self.navbar_active = !self.navbar_active;
                true
            }

            Search(value) => {
                log::info!("enter input the input is {:?}", value);
                self.search(value);
                true
            }
            ChangeSearchCat(cat) => {
                // clear input
                let mut inp = self.get_input();
                inp.set_value("");
                // change search_placeholder
                let ph = match cat {
                    Category::Symptom => { "肩背痛" }
                    Category::Disease => { "皮肤炎症" }
                    Category::Drug => { "undefined" }
                    Category::Department => { "undefined" }
                    Category::Check => { "undefined" }
                };
                self.search_placeholder = ph.to_string();

                self.search_cat = cat;
                true
            }
            FillPlaceholder => {
                let mut inp = self.get_input();
                if inp.value().len() == 0 {
                    inp.set_value(inp.placeholder().as_str());
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn changed(&mut self, ctx: &Context<Self>) -> bool {
        false
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <>
                { self.view_nav(ctx.link()) }

                <main>
                    <Router<Route> render={Router::render(switch)} />
                </main>

                { self.view_footer() }
            </>
        }
    }
}

// view function
impl App {
    fn view_nav(&self, link: &Scope<Self>) -> Html {
        let Self { navbar_active, .. } = *self;
        let active_class = if navbar_active { "is-active" } else { "" };

        let onkeypress = link.batch_callback(|e: KeyboardEvent| {
            if e.key() == "Enter" {
                let input = e
                    .target()
                    .and_then(|t| t.dyn_into::<HtmlInputElement>().ok());
                let value = input.map(|x| x.value());
                // input.set_value("");

                value.map(AppMsg::Search)
            } else {
                log::debug!("search input -> event: {:?}, enter key: {:?}", e, e.key());
                None
            }
        });

        let fill_placeholder = link.callback(|_| {
            AppMsg::FillPlaceholder
        }
        );


        html! {
            <section class="navbar">
                <nav class="navbar" role="navigation" aria-label="main navigation">
                    <div class="navbar-brand">
                        <a class="navbar-item" href="https://bulma.io">
                            <img src="https://bulma.io/images/bulma-logo.png" width="112" height="28" />
                        </a>

                        <a role="button" class={classes!("navbar-burger", "burger" , active_class)} aria-label="menu" aria-expanded="false" onclick={link.callback(|_| AppMsg::ToggleNavbar)}>
                            <span aria-hidden="true"></span>
                            <span aria-hidden="true"></span>
                            <span aria-hidden="true"></span>
                        </a>
                    </div>


                    <div class={classes!("navbar-menu", active_class)}>
                        <div class="navbar-start">

                            <Link<Route> classes={classes!("navbar-item")} route={Route::Home}>
                                { "Home" }
                            </Link<Route>>

                            // <a class="navbar-item">
                                // { "Home" }
                                // </a>

                            <Link<Route> classes={classes!("navbar-item")} route={Route::Symptom}>
                                { "症状" }
                            </Link<Route>>

                        </div>

                        <div class="navbar-end">
                            { self.view_nav_cats(link) }

                             <div class="navbar-item">
                                <div class="field">
                                    <p class="control has-icons-right">
                                        // <input ref={self.search_inp.clone()} class="input is-success" type="text" placeholder="肩背痛" onkeypress={onkeypress} />
                                        <input ref={self.search_inp.clone()} class="input is-success" type="text" placeholder={self.search_placeholder.clone()} onkeypress={onkeypress} onclick={fill_placeholder} />
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

    fn view_nav_cats(&self, link: &Scope<Self>) -> Html {
        html! {
            <div class="navbar-item has-dropdown is-hoverable">
                <a class="navbar-link is-arrowless">
                    <div class="field has-addons">
                        <span class="tag is-warning is-light"> { self.search_cat.clone() }  </span>
                    </div>
                </a>

                <div class="navbar-dropdown">
                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(Category::Symptom))} >
                        { "Symptom" }
                    </a>

                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(Category::Disease))} >
                        { "Disease" }
                    </a>
                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(Category::Drug))} >
                        { "Drug" }
                    </a>
                    // <hr class="navbar-divider" onclick={link.callback(|_| AppMsg::ChangeSearchCat(Category::Department))} >
                    <a class="navbar-item">
                        { "Department" }
                    </a>
                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(Category::Check))} >
                        { "Check" }
                    </a>
                </div>
            </div>
        }
    }

    fn view_footer(&self) -> Html {
        html! {
            <footer class="footer">
                <div class="content has-text-centered">
                    { "Powered by " }
                    <a href="https://yew.rs">{ "Yew" }</a>
                    { " using " }
                    <a href="https://bulma.io">{ "Bulma" }</a>
                </div>
            </footer>
        }
    }
}

// method
impl App {
    fn get_input(&self) -> HtmlInputElement {
        self.search_inp.cast::<HtmlInputElement>().unwrap()
    }
    fn search(&mut self, name: String) {
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
        log::debug!("search cat: {}", self.search_cat);
        unsafe {
            spawn_local(bindings::display_network(self.search_cat.to_string(), name));
        }
    }
}

fn switch(routes: &Route) -> Html {
    match routes {
        Route::Home => {
            html! { <Home /> }
        }
        Route::Symptom => {
            html! { <PageSymptom /> }
        }
        Route::NotFound => {
            html! { <PageNotFound /> }
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

    // unsafe {
    //     bindings::main();
    // }

    // only required for stdweb
    // yew::run_loop();

    yew::start_app_in_element::<App>(mount_point);
    log::info!("after start app");
}
