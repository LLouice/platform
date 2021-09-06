use wasm_bindgen_futures::{spawn_local, JsFuture};
use wasm_logger;

use yew::prelude::*;

use ybc::NavbarFixed::Top;
use ybc::TileCtx::{Ancestor, Child, Parent};
use ybc::TileSize::Four;

use crate::bindings;

struct App; // An application component.

impl Component for App {
    type Message = ();
    type Properties = ();

    fn create(_props: Self::Properties, _link: ComponentLink<Self>) -> Self {
        Self {}
    }

    fn update(&mut self, _msg: Self::Message) -> ShouldRender {
        false
    }

    fn change(&mut self, _props: Self::Properties) -> ShouldRender {
        false
    }

    fn view(&self) -> Html {
        html! {
            <>

                <div class="block">
                    <div class="network container">
                        <div id="network" style="width:1200px;height:1000px;margin:auto"></div>
                    </div>

                    <div class="stats_rels container">
                        <div id="stats_rels" style="width:1200px;height:1000px;margin:auto"></div>
                    </div>
                </div>

            </>
        }
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
