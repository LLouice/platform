use web_sys::HtmlElement;
use yew::prelude::*;

use crate::app::{App, NodeInfo,};

pub struct AI;


#[derive(Properties, Clone, PartialEq)]
pub struct AIProps {
    pub node_info: NodeInfo,
}

impl Component for AI {
    type Message = ();
    type Properties = AIProps;

    fn create(ctx: &Context<Self>) -> Self {
        log::info!("AI page created, node_info: {:?}", ctx.props().node_info);
        Self
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <>
                // { Self::view_hero() }
                { self.view_table(ctx)}
            </>
        }
    }

    fn rendered(&mut self, ctx: &Context<Self>, first_render: bool) {
        log::info!("AI prediction rendered...");
    }
}

impl AI {
    // written during testing
    #[allow(dead_code)]
    fn view_hero() -> Html {
        html! {
            <section class="hero is-info is-bold is-large">
                <div class="hero-body">
                    <div class="container">
                        <h1 class="title">
                            { "AI" }
                        </h1>
                        <h2 class="subtitle">
                            { "预测" }
                        </h2>
                    </div>
                </div>
            </section>
        }
    }

    fn view_table(&self, ctx: &Context<Self>) -> Html {
        html! {
            <section class="hero is-info is-bold is-large">
                <div class="hero-body">
                    <div class="container">
                        <h1 class="title">
                            { "AI" }
                        </h1>
                        <h2 class="subtitle">
                            // { format!("预测 {self.label}::{self.name}") }
                            { ctx.props().node_info.clone() }
                        </h2>
                    </div>
                </div>
            </section>
        }
    }
}
