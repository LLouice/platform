use web_sys::HtmlElement;
use yew::prelude::*;

use platform::data::QRandomSample;

use crate::app::App;

pub struct PageCategory {
    word_cloud: NodeRef,
}


#[derive(Properties, Clone, PartialEq)]
pub struct PageCategoryProps {
    pub query: QRandomSample,
}

impl Component for PageCategory {
    type Message = ();
    type Properties = PageCategoryProps;

    fn create(_ctx: &Context<Self>) -> Self {
        Self { word_cloud: NodeRef::default() }
    }

    fn view(&self, _ctx: &Context<Self>) -> Html {
        html! {
            <>
                { self.view_chart() }
            </>
        }
    }

    fn rendered(&mut self, ctx: &Context<Self>, first_render: bool) {
        // first_render, auto load data, no interactive event emmit
        if first_render {
            if self.word_cloud.cast::<HtmlElement>().is_some() {
                log::info!("the div exists!");
                App::display_word_cloud(ctx.props().query);
            } else {
                log::info!("the div not exists!");
            }
        }
    }
}


impl PageCategory {
    fn view_chart(&self) -> Html {
        log::info!("[category] enter veiw_chart");
        let node = html! {
            <div class="block">
                <div class="container categor_word_cloud">
                    <div ref={self.word_cloud.clone()} id="category_word_cloud" style="width:90%;height:calc(100vh);"></div>
                </div>
            </div>
        };
        node
    }


    #[allow(dead_code)]
    fn view_hero(&self) -> Html {
        html! {
            <section class="hero is-info is-bold is-large">
                <div class="hero-body">
                    <div class="container">
                        <h1 class="title">
                            { "Symptom" }
                        </h1>
                        <h2 class="subtitle">
                            { "症状云" }
                        </h2>
                    </div>
                </div>
            </section>
        }
    }
}
