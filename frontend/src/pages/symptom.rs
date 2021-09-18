use yew::prelude::*;

use crate::app::App;

pub struct PageSymptom;

impl Component for PageSymptom {
    type Message = ();
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, _ctx: &Context<Self>) -> Html {
        html! {
            <>
                { self.view_chart() }
            </>
        }
    }
}


impl PageSymptom {
    fn view_chart(&self) -> Html {
        let node = html! {
            <div class="block">
                <div class="container symptom_word_cloud">
                    <div id="symptom_word_cloud" style="width:90%;height:calc(100vh);"></div>
                </div>
            </div>
        };
        App::display_symptom_word_cloud();
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
