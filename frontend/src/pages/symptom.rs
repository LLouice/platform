use yew::prelude::*;

pub struct PageSymptom;

impl Component for PageSymptom {
    type Message = ();
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, _ctx: &Context<Self>) -> Html {
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
