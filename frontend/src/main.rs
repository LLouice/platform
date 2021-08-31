use ybc::NavbarFixed::Top;
use ybc::TileCtx::{Ancestor, Child, Parent};
use ybc::TileSize::Four;
use yew::prelude::*;

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
                <ybc::Button classes=classes!("is-primary")> { "primary" } </ybc::Button>
                <ybc::Button classes=classes!("is-link")> { "link" } </ybc::Button>
                <ybc::Button classes=classes!("is-link") loading=true> { "loading" } </ybc::Button>

                <div class="block">
                    <h1 class="is-warning is-large"> { "something" } </h1>
                    <h1 class="is-warning is-size-2"> { "stuff" } </h1>
                </div>

                <div class="columns">
                    <div class="column">
                        { "First column" }
                    </div>

                    <div class="column">
                        { "Second column" }
                    </div>

                    <div class="column">
                        { "Third column" }
                    </div>

                    <div class="column is-primary is-uppercase">
                        { "Four column" }
                    </div>
                </div>

                // main canvas
                <div id="main.sub"> </div>
                <div id="main" style="width:1200px;height:1000px;margin:auto"></div>

            </>
        }
    }
}

fn main() {
    env_logger::init();

    yew::start_app::<App>();
}
