use anyhow::{anyhow, bail, Result};
use js_sys::Function;
use reqwasm::http::Request;
use wasm_bindgen::{closure::Closure, JsCast, JsValue};
use wasm_bindgen_futures::spawn_local;
use wasm_logger;
use web_sys::{Element, HtmlElement, HtmlInputElement};
use yew::html::Scope;
use yew::prelude::*;
use yew::utils::{document, window};
use yew_router::prelude::*;
use yew_router::push_route;

pub(crate) use platform::data::{NodeInfo, NodeLabel, QRandomSample};

use crate::{into_js_fn, into_js_fn_mut, js_apply, js_get, js_set};
use crate::bindings;
use crate::pages::{category::PageCategory, home::Home, ai::AI, page_not_found::PageNotFound};

#[derive(Debug, Default)]
pub(crate) struct App {
    search_inp: NodeRef,
    search_cat: NodeLabel,
    search_value: Option<String>,
    search_placeholder: String,
    navbar_active: bool,
    route_cat: NodeLabel,
}

#[derive(Routable, PartialEq, Clone, Debug)]
pub enum Route {
    // #[at("/posts/:id")]
    // Post { id: u64 },
    #[at("/")]
    Home,

    #[at("/:node_info")]
    Network { node_info: NodeInfo },

    #[at("/category/:query")]
    Category { query: QRandomSample },

    #[at("/ai/:node_info")]
    AI { node_info: NodeInfo },

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
    ChangeSearchCat(NodeLabel),
    FillPlaceholder,
    ChangeRouteCat(NodeLabel),
    RefreshSample,
    AIPredict,
}

impl Component for App {
    type Message = AppMsg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            search_cat: NodeLabel::Symptom,
            search_placeholder: String::from("čŠčį"),
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
                self.search_value = Some(value.clone());
                self.search(value);
                true
            }
            ChangeSearchCat(cat) => {
                // clear input
                let inp = self.get_input();
                inp.set_value("");
                // change search_placeholder
                let ph = match cat {
                    NodeLabel::Symptom => "čŠčį",
                    NodeLabel::Disease => "įŽč¤įį",
                    NodeLabel::Drug => "undefined",
                    NodeLabel::Department => "undefined",
                    NodeLabel::Check => "undefined",
                    NodeLabel::Area => "undefined",
                    _ => "undefined",
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
            ChangeRouteCat(cat) => {
                log::info!("ChangeRouteCat");
                self.route_cat = cat;
                let query = QRandomSample {
                    label: self.route_cat,
                    limit: Some(10),
                };
                App::display_word_cloud(query);
                true
            }
            RefreshSample => {
                log::info!("RefreshSample");
                let query = QRandomSample {
                    label: self.route_cat,
                    limit: Some(10),
                };
                App::display_word_cloud(query);
                true
            }
            AIPredict => {
                let inp = self.get_input();
                if inp.value().len() == 0 {
                    inp.set_value(inp.placeholder().as_str());
                }

                push_route(Route::AI {
                    node_info: NodeInfo {
                        label: self.search_cat,
                        name: inp.value(),
                    }
                }
                );
                true

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

        let fill_placeholder = link.callback(|_| AppMsg::FillPlaceholder);

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

                            <Link<Route> classes={classes!("navbar-item")}
                                         route={Route::Network{ node_info: NodeInfo {label: self.search_cat,
                                                             name: if let Some(v) = &self.search_value
                                                                    { v.clone()} else {self.search_placeholder.clone()}
                                         }} }> { "Home" }
                            </Link<Route>>

                            // <a class="navbar-item">
                                // { "Home" }
                                // </a>

                            // <Link<Route> classes={classes!("navbar-item")} route={Route::Category} >
                            //     <span class="is-danger" ondblclick={refresh_sample}>
                            //         { "įįļ" }
                            //     </span>
                            // </Link<Route>>

                            { self.view_nav_cats(link) }

                        </div>

                        <div class="navbar-end">
                            { self.view_nav_search_cats(link) }


                             <div class="navbar-item">
                                <div class="field">
                                    <p class="control has-icons-right">
                                        // <input ref={self.search_inp.clone()} class="input is-success" type="text" placeholder="čŠčį" onkeypress={onkeypress} />
                                        <input ref={self.search_inp.clone()} class="input is-success" type="text" placeholder={self.search_placeholder.clone()} onkeypress={onkeypress} onclick={fill_placeholder} />
                                        <span class="icon is-small is-right">
                                            <i class="fas fa-search"></i>
                                        </span>
                                    </p>
                                </div>
                            </div>

                            { self.view_nav_ai_prediction(link) }

                        </div>
                    </div>
                </nav>
            </section>
        }
    }

    fn view_nav_cats(&self, link: &Scope<Self>) -> Html {
        let refresh_sample = link.callback(|_| AppMsg::RefreshSample);

        html! {
            <div class="navbar-item has-dropdown is-hoverable">
                // <a class="navbar-link is-arrowless">
                //     <div class="field has-addons">
                //         <span class="is-warning is-light" ondblclick={refresh_sample}> { self.route_cat }  </span>
                //     </div>
                // </a>


                <Link<Route> classes={classes!("navbar-link", "is-arrowless")} route={Route::Category{
                             query: QRandomSample {
                                label: self.route_cat,
                                limit: Some(10),
                             }}} >
                    <div class="field has-addons">
                        // <span class="is-warning is-light" ondblclick={refresh_sample.clone()} > { self.route_cat }  </span>
                        <span onclick={refresh_sample.clone()} ondblclick={refresh_sample.clone()} > { self.route_cat } </span>
                    </div>
                </Link<Route>>


                // <div classes={classes!("navbar-link", "is-arrowless")}>
                //     <div class="field has-addons">
                //         <span class="is-warning is-light" ondblclick={refresh_sample}> { self.route_cat }  </span>
                //     </div>
                // </div>

                // <div class="navbar-dropdown">
                //     <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Symptom))} >
                //         { "Symptom" }
                //     </a>
                //
                //     <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Disease))} >
                //         { "Disease" }
                //     </a>
                //     <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Drug))} >
                //         { "Drug" }
                //     </a>
                //     // <hr class="navbar-divider" onclick={link.callback(|_| AppMsg::ChangeSearchCat(Category::Department))} >
                //     <a class="navbar-item">
                //         { "Department" }
                //     </a>
                //     <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Check))} >
                //         { "Check" }
                //     </a>
                //     <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Area))} >
                //         { "Area" }
                //     </a>
                // </div>


                <div class="navbar-dropdown">
                    <Link<Route> classes={classes!("navbar-item")} route={Route::Category{
                                 query: QRandomSample {
                                    label: NodeLabel::Symptom,
                                    limit: Some(10),
                                 }}} >
                        <span onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Symptom))} ondblclick={refresh_sample.clone()} > { "Symptom" }  </span>
                    </Link<Route>>

                    <Link<Route> classes={classes!("navbar-item")} route={Route::Category{
                                 query: QRandomSample {
                                    label: NodeLabel::Disease,
                                    limit: Some(10),
                                 }}} >
                        <span onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Disease))} ondblclick={refresh_sample.clone()} > { "Disease" }  </span>
                    </Link<Route>>

                    <Link<Route> classes={classes!("navbar-item")} route={Route::Category{
                                 query: QRandomSample {
                                    label: NodeLabel::Drug,
                                    limit: Some(10),
                                 }}} >
                        <span onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Drug))} ondblclick={refresh_sample.clone()} > { "Drug" }  </span>
                    </Link<Route>>

                    <Link<Route> classes={classes!("navbar-item")} route={Route::Category{
                                 query: QRandomSample {
                                    label: NodeLabel::Department,
                                    limit: Some(10),
                                 }}} >
                        <span onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Department))} ondblclick={refresh_sample.clone()} > { "Department" }  </span>
                    </Link<Route>>

                    <Link<Route> classes={classes!("navbar-item")} route={Route::Category{
                                 query: QRandomSample {
                                    label: NodeLabel::Check,
                                    limit: Some(10),
                                 }}} >
                        <span onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Check))} ondblclick={refresh_sample.clone()} > { "Check" }  </span>
                    </Link<Route>>

                    <Link<Route> classes={classes!("navbar-item")} route={Route::Category{
                                 query: QRandomSample {
                                    label: NodeLabel::Area,
                                    limit: Some(10),
                                 }}} >
                        <span onclick={link.callback(|_| AppMsg::ChangeRouteCat(NodeLabel::Area))} ondblclick={refresh_sample} > { "Area" }  </span>
                    </Link<Route>>
                </div>
            </div>
        }
    }

    fn view_nav_search_cats(&self, link: &Scope<Self>) -> Html {
        html! {
            <div class="navbar-item has-dropdown is-hoverable">
                <a class="navbar-link is-arrowless">
                    <div class="field has-addons">
                        <span class="tag is-warning is-light"> { self.search_cat.clone() }  </span>
                    </div>
                </a>

                <div class="navbar-dropdown">
                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(NodeLabel::Symptom))} >
                        { "Symptom" }
                    </a>

                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(NodeLabel::Disease))} >
                        { "Disease" }
                    </a>
                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(NodeLabel::Drug))} >
                        { "Drug" }
                    </a>
                    // <hr class="navbar-divider" onclick={link.callback(|_| AppMsg::ChangeSearchCat(Category::Department))} >
                    <a class="navbar-item">
                        { "Department" }
                    </a>
                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(NodeLabel::Check))} >
                        { "Check" }
                    </a>
                    <a class="navbar-item" onclick={link.callback(|_| AppMsg::ChangeSearchCat(NodeLabel::Area))} >
                        { "Area" }
                    </a>
                </div>
            </div>
        }
    }

    fn view_nav_ai_prediction(&self, link: &Scope<Self>) -> Html {
        let predict = link.callback(|_| AppMsg::AIPredict);
        html! {
            <a role="button" class={classes!("navbar-item", "is-bold")} onclick={predict}>{ "AI" }</a>
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
        //         "http://localhost:9090/get_out_links?label=Symptom&name={}",
        //         name
        //     );

        //     let resp = Request::get(&uri).send().await.unwrap();
        //     log::info!("{:?}", resp);
        //     let graph_data: GraphData = resp.json().await.unwrap();
        //     log::info!("graph_data: {:?}", graph_data);
        // });
        log::debug!("search cat: {}", self.search_cat);
        // spawn_local(bindings::display_network(self.search_cat.to_string(), name));
        // insert slider
        log::info!("Switch into Route::Network...");
        // Switch route
        push_route(Route::Network {
            node_info: NodeInfo {
                label: self.search_cat,
                name,
            }
        }
        );
    }
}

// function
impl App {
    pub async fn get_word_cloud_data(query: QRandomSample) -> Result<String> {
        let QRandomSample { label, limit } = query;
        let limit = limit.unwrap_or(10);
        let uri = format!(
            "http://localhost:9090/random_sample?label={}&limit={}",
            label, limit
        );

        let resp = Request::get(&uri).send().await.unwrap();
        // log::info!("{:?}", resp);
        let text = resp.text().await?;

        log::debug!("api return text: {}", text);

        if resp.ok() {
            Ok(text)
        } else {
            bail!(text)
        }
    }

    pub fn display_word_cloud(query: QRandomSample) {
        // get data from server
        spawn_local(async move {
            log::info!("execute display_word_cloud!");
            let data = Self::get_word_cloud_data(query).await.map_err(|e| {
                log::error!("{}", e);
                e
            });

            let data = if let Ok(data) = data {
                js_sys::JSON::parse(data.as_str())
                    .map(|ref x| js_sys::Array::from(x))
                    .ok()
            } else {
                None
            };
            bindings::display_word_cloud(data);
        });
    }

    pub fn display_main(label: Option<NodeLabel>, name: Option<String>, first_render: bool) {
        spawn_local(async move {
            bindings::main(label.map(|x| x.to_string()), name).await;
            if first_render {
                App::insert_slider();
            }
        });
    }

    fn insert_slider() {
        let _ = Self::_insert_slider().map_err(|e| {
            log::error!("{:?}", e);
        });
    }

    fn _insert_slider() -> std::result::Result<(), JsValue> {
        log::info!("insert slider...");
        let doc = document();
        let network_svg = doc.query_selector("#network svg")?;
        log::debug!("network_svg: {:?}", network_svg);
        if let Some(network_svg) = network_svg {
            log::debug!("network_svg exists");
            // svg position
            let svg_rec = network_svg
                .unchecked_ref::<Element>()
                .get_bounding_client_rect();
            // from left client
            let svg_left: f64 = svg_rec.left();
            let svg_width: f64 = svg_rec.width();
            let svg_height: f64 = svg_rec.height();
            let svg_top: f64 = svg_rec.top();
            let svg_right: f64 = svg_rec.right();

            log::info!("[svg] left: {:?}, width: {:?}, height: {:?}, top: {:?}, right: {:?}",
                svg_left,
                svg_width,
                svg_height,
                svg_top,
                svg_right,
            );

            const SVGNS: Option<&str> = Some("http://www.w3.org/2000/svg"); // svg namespace
            let slider: HtmlElement = doc.create_element_ns(SVGNS, "g")?.unchecked_into();
            let line = doc.create_element_ns(SVGNS, "line")?;
            // line position
            // the ratio margin right of svg
            let mr_ratio = 0.05;
            // the ratio or line len
            let len_ratio = 0.1;
            // the ratio from the top of svg
            let mt_ratio = 0.15;

            let x1 = svg_width * (1. - mr_ratio - len_ratio);
            let y1 = svg_height * mt_ratio;
            let line_len = svg_width * len_ratio;
            let line_half = line_len / 2.;
            let x2 = x1 + line_len;
            let r = 5_f64;
            let c = x1 + line_half;
            let line_half_r = line_half - r;

            let get_ratio = move |pos: f64| {
                let x = 1. + (pos - c) / line_half_r;
                if x > 0. {
                    x
                } else {
                    0.
                }
            };

            log::info!("x1: {:?}, x2: {:?}, c: {:?}", x1, x2, c);

            // set line attributes
            line.set_attribute_ns(None, "x1", x1.to_string().as_str())?;
            line.set_attribute_ns(None, "y1", y1.to_string().as_str())?;
            line.set_attribute_ns(None, "x2", x2.to_string().as_str())?;
            line.set_attribute_ns(None, "y2", y1.to_string().as_str())?;
            line.set_attribute_ns(None, "stroke", "#45c589")?;
            line.set_attribute_ns(None, "stroke-width", r.to_string().as_str())?;
            line.set_attribute_ns(None, "id", "sliderLine")?;
            slider.append_child(&line)?;
            // dot
            let dot: HtmlElement = doc.create_element_ns(SVGNS, "circle")?.unchecked_into();
            dot.set_attribute_ns(None, "r", r.to_string().as_str())?;
            dot.set_attribute_ns(None, "transform", &format!("translate({} {})", c, y1))?;
            dot.set_attribute_ns(None, "id", "dot")?;

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
                let dot: HtmlElement = doc.query_selector("#dot").unwrap().unwrap().unchecked_into();

                // init info
                // for calc move distance
                let start_ptr_x: i32 = e.client_x();
                // the dot center init x
                // absolute position
                let init_x: f64 = dot.unchecked_ref::<Element>().get_bounding_client_rect().x() - svg_left + r;

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
                    let _x2 = x2 - r;
                    if new_x > _x2 {
                        new_x = _x2;
                    }
                    log::debug!("new_x: {:?}", new_x);
                    let _ = js_set!(&dot, "transform", "baseVal", "0", "matrix" => "e", new_x).map_err(|e| log::error!("{:?}", e));
                    let dot_attr_e = js_get!(&dot, "transform", "baseVal", "0", "matrix", "e");
                    log::info!("dot_attr_e: {:?}", dot_attr_e);

                    let ratio = get_ratio(new_x);
                    let pre_ratio = js_get!(&dot, "ratio");
                    let pre_ratio: f64 = match pre_ratio {
                        Ok(x) => {x.as_f64().unwrap_or(1.0)}
                        Err(_) => {1.0}
                    };
                    let _ = Self::change_symbol_size(pre_ratio, ratio).map_err(|e| log::error!("{:?}", e));
                    // save current ratio in dot for getting the origin symbolSize
                    let _ = js_set!(&dot, "ratio", ratio);

                    log::debug!("current ratio: {:?}", ratio);
                }) as Box<dyn FnMut()>).into_js_value().into();

                doc.add_event_listener_with_callback("mousemove", &on_mouse_move).expect("error on add mousemove listener");

                // generate on fly
                let on_mouse_move_c  = on_mouse_move.clone();
                // run many time, it remove listener
                let on_mouse_up: Function = Closure::once_into_js(move || {
                    document().remove_event_listener_with_callback("mousemove", &on_mouse_move_c).expect("error on remove mousemove listener");
                }).into();

                // should use AddEventListenerOptions.once to free mouseup on fly
                doc.add_event_listener_with_callback_and_add_event_listener_options("mouseup", &on_mouse_up,
                   web_sys::AddEventListenerOptions::new().once(true)
                ).expect("error on add mouseup listener");
            }).into();

            dot.set_onmousedown(Some(&on_mousedown));
            dot.set_ondrag(None);

            slider.append_child(&dot)?;

            let dblclick: Function = into_js_fn!(move || {
                log::info!("on dblclick");
                let _e: MouseEvent = window().event().unchecked_into();
                let dot = document().query_selector("#dot").unwrap().unwrap();
                let _ = js_set!(&dot, "transform", "baseVal", "0", "matrix" => "e", c)
                    .map_err(|e| log::error!("{:?}", e));
                // reset ratio and symbolSize
                let pre_ratio = js_get!(&dot, "ratio");
                let pre_ratio: f64 = match pre_ratio {
                    Ok(x) => x.as_f64().unwrap_or(1.0),
                    Err(_) => 1.0,
                };
                let _ =
                    Self::change_symbol_size(pre_ratio, 1.0).map_err(|e| log::error!("{:?}", e));
                let _ = js_set!(&dot, "ratio", 1.0);
            });
            slider.set_ondblclick(Some(&dblclick));
            network_svg.append_child(&slider)?;

            log::debug!("insert slider done!");

            Ok(())
        } else {
            Err(JsValue::from_str("network svg is not exists now"))
        }
    }

    // call by update?
    fn change_symbol_size(pre_ratio: f64, ratio: f64) -> std::result::Result<(), JsValue> {
        let chart_opt = dbg!(bindings::get_echarts_option(None));
        log::debug!("chart_opt: {:?}", chart_opt);
        let chart = dbg!(js_get!(&chart_opt, "chart"))?;
        let opt = js_get!(&chart_opt, "opt")?;
        let opt_data = dbg!(js_get!(&opt, "series", "0", "data"))?;
        log::debug!("char: {:?}", chart);
        log::debug!("opt: {:?}", opt_data);

        // iterate data to change it's SymbolSize
        let it = js_sys::try_iter(&opt_data)?.ok_or_else(|| "need to pass iterable JS values!")?;

        for x in it {
            // If the iterator's `next` method throws an error, propagate it
            // up to the caller.
            let x = x?;

            // If `x` is a number, add it to our array of numbers!
            let symbol_size = js_get!(&x, "symbolSize")?
                .as_f64()
                .ok_or("no symbolSize in Option data")?;
            log::debug!("ratio: {}, new symbolSize {:?}", ratio, symbol_size * ratio);
            let _ = js_set!(&x, "symbolSize", symbol_size / pre_ratio * ratio);
            log::debug!("{:?}", js_get!(&x, "symbolSize"));
        }
        // reset option to trigger echarts update
        js_apply!(&chart, "setOption", [&opt])?;

        Ok(())
    }
}

fn switch(routes: &Route) -> Html {
    match routes {
        Route::Home => {
            html! { <Home label={let x: Option<NodeLabel> = None; x} name={let x: Option<String> = None; x} /> }
        }
        Route::Network { node_info: NodeInfo { label, name } } => {
            log::debug!("in switch route Network, label: {:?} name: {:?}", label, name);
            html! { <Home label={Some(*label)} name={Some(name.clone())} /> }
        }
        Route::Category { query } => {
            html! { <PageCategory query={*query}/> }
        }

        Route::AI { node_info } => {
            html! { <AI node_info={node_info.clone()} /> }
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

    // ---- debug ----

    log::info!("after start app");
}
