use js_sys::Function;
use wasm_bindgen::{closure::Closure, JsCast, JsValue};
use wasm_bindgen_futures::spawn_local;
use wasm_logger;
use web_sys::{Element, HtmlElement, HtmlInputElement};
use yew::html::Scope;
use yew::prelude::*;
use yew::utils::{document, window};
use yew_router::prelude::*;

use crate::{into_js_fn, into_js_fn_mut, js_get, js_set};
use crate::bindings;
use crate::pages::{home::Home, page_not_found::PageNotFound, symptom::PageSymptom};

#[derive(Debug, Default)]
pub(crate) struct App {
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
    #[allow(dead_code)]
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

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
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
                let inp = self.get_input();
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
                let inp = self.get_input();
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

    fn changed(&mut self, _ctx: &Context<Self>) -> bool {
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
        spawn_local(bindings::display_network(self.search_cat.to_string(), name));
        // insert slider
        log::info!("insert slider...");
        Self::insert_slider();
    }
}


// function
impl App {
    pub fn display_main() {
        spawn_local(async {
            bindings::main().await;
            App::insert_slider();
        });
    }

    fn insert_slider() {
        let _ = Self::_insert_slider().map_err(|e| {
            log::error!("{:?}", e);
        });
    }

    // FIXME: use ? instead of unwrap, expect
    fn _insert_slider() -> std::result::Result<(), JsValue> {
        log::info!("insert slider...");
        let doc = document();
        let win = window();
        let network_svg = doc.query_selector("#network svg")?;
        log::debug!("network_svg: {:?}", network_svg);
        // FIXME: timeout retry
        if let Some(network_svg) = network_svg {
            log::debug!("network_svg exists");
            const SVGNS: Option<&str> = Some("http://www.w3.org/2000/svg"); // svg namespace
            let slider: HtmlElement = doc.create_element_ns(SVGNS, "g").unwrap().unchecked_into();
            let line = doc.create_element_ns(SVGNS, "line").unwrap();
            let width: f64 = win.inner_width().unwrap().as_f64().expect("error on get window inner_width");
            let height: f64 = win.inner_height().unwrap().as_f64().expect("error on get window inner_width");
            // line position
            let x1 = width * 0.75;
            let y1 = height * 0.75;
            let line_len = width * 0.1;
            let x2 = x1 + line_len;
            let r = 5_f64;
            let c = x1 + line_len / 2.0;

            log::info!("x1: {:?}, x2: {:?}, c: {:?}", x1, x2, c);

            // set line attributes
            line.set_attribute_ns(None, "x1", x1.to_string().as_str()).unwrap();
            line.set_attribute_ns(None, "y1", y1.to_string().as_str()).unwrap();
            line.set_attribute_ns(None, "x2", x2.to_string().as_str()).unwrap();
            line.set_attribute_ns(None, "y2", y1.to_string().as_str()).unwrap();
            line.set_attribute_ns(None, "stroke", "#45c589").unwrap();
            line.set_attribute_ns(None, "stroke-width", r.to_string().as_str()).unwrap();
            line.set_attribute_ns(None, "id", "sliderLine").unwrap();
            slider.append_child(&line).expect("error on slider append line child");
            // dot
            let dot: HtmlElement = doc.create_element_ns(SVGNS, "circle").unwrap().unchecked_into();
            dot.set_attribute_ns(None, "r", r.to_string().as_str()).unwrap();
            dot.set_attribute_ns(None, "transform", &format!("translate({} {})", c, y1)).unwrap();
            dot.set_attribute_ns(None, "id", "dot").unwrap();

            log::debug!("set dot attribute done!");

            // the dot target attr
            let dot_attr_e = js_get!(&dot, "transform", "baseVal", "0", "matrix", "e")?;
            log::info!("dot_attr_e: {:?}", dot_attr_e);

            // drag
            // here move but not consume value
            let on_mousedown: Function = into_js_fn_mut!(move || {
                let e: MouseEvent = window().event().unchecked_into();
                e.prevent_default();
                let doc = document();
                let line: HtmlElement = doc.query_selector("#sliderLine").unwrap().unwrap().unchecked_into();
                let dot: HtmlElement = doc.query_selector("#dot").unwrap().unwrap().unchecked_into();

                // init info
                let line_left = line.unchecked_into::<Element>().get_bounding_client_rect().x();
                // svg margin left: for give transform value in svg
                let svg_ml: f64 = line_left - x1;
                log::debug!("line left: {}", line_left);
                log::debug!("svg_ml: {}", svg_ml);

                // for calc move distance
                let start_ptr_x: i32 = e.client_x();
                // the dot center init x
                let init_x: f64 = dot.unchecked_ref::<Element>().get_bounding_client_rect().x() - svg_ml + r;

                log::debug!("x: {:?}, start_ptr_x: {:?}, init_x: {:?}, screen_x: {:?}, offset_x: {:?}, client_left: {:?}, offset_left: {:?}",
                e.x(),
                    start_ptr_x,
                    init_x,
                    e.screen_x(),
                    e.offset_x(),
                    dot.client_left(),
                    dot.offset_left(),
                );

                // generate on fly
                let on_mouse_move: Function = Closure::wrap(Box::new(move || {
                    log::info!("on mouse move");
                    let e: MouseEvent = window().event().unchecked_into();
                    let move_distance: i32 = e.client_x() - start_ptr_x;
                    let mut new_x = init_x + move_distance as f64;
                    log::debug!("client_x: {:?}, move_distance: {:?}, new_x: {:?}", e.client_x(), move_distance, new_x);
                    let _x1 = x1 + r;
                    if new_x < _x1 {
                        new_x = _x1;
                    }
                    let _x2 = x2 - 2. * r;
                    if new_x > _x2 {
                        new_x = _x2;
                    }
                    log::debug!("new_x: {:?}", new_x);
                    let _ = js_set!(&dot, "transform", "baseVal", "0", "matrix" => "e", new_x).map_err(|e| log::error!("{:?}", e));
                    let dot_attr_e = js_get!(&dot, "transform", "baseVal", "0", "matrix", "e");
                    log::info!("dot_attr_e: {:?}", dot_attr_e);

                }) as Box<dyn FnMut()>).into_js_value().into();

                doc.add_event_listener_with_callback("mousemove", &on_mouse_move).expect("error on add mousemove listener");

                // generate on fly
                // only call once,otherwise throw a exception, so don't need remove self(can't do it in rust actually)
                let on_mouse_move_c  = on_mouse_move.clone();
                // run many time, it remove listener
                let on_mouse_up: Function = Closure::once_into_js(move || {
                    document().remove_event_listener_with_callback("mousemove", &on_mouse_move_c).expect("error on remove mousemove listener");
                }).into();

                // here on_mouse_up will free after on_mouse_up is called
                doc.add_event_listener_with_callback_and_add_event_listener_options("mouseup", &on_mouse_up,
                   web_sys::AddEventListenerOptions::new().once(true)
                ).expect("error on add mouseup listener");
            }).into();

            dot.set_onmousedown(Some(&on_mousedown));
            dot.set_ondrag(None);

            slider.append_child(&dot).expect("error on slider append dot child");

            let on_dbclick: Function = into_js_fn!(move ||{
                log::info!("on dblclick");
                let _e: MouseEvent = window().event().unchecked_into();
                let dot = document().query_selector("#dot").unwrap().unwrap();
                let _ = js_set!(&dot, "transform", "baseVal", "0", "matrix" => "e", c).map_err(|e| log::error!("{:?}", e));
                // dot.set_attribute_ns(None, "transform", &format!("translate({} {})", c, y1)).unwrap();
            });
            slider.set_ondblclick(Some(&on_dbclick));
            network_svg.append_child(&slider).expect("error on append slider");

            log::debug!("insert slider done!");

            Ok(())
        } else {
            Err(JsValue::from_str("network svg is not exists now"))
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
    //  echarts
    // run in next microtask tick
    // App::display_main();
    // let res = App::insert_slider();
    // log::debug!("insert res: {:?}", res);

    log::info!("after start app");
}
