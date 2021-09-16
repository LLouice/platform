use yew::prelude::*;

use crate::app::App;

pub struct Home;

impl Component for Home {
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



impl Home {
    #[allow(dead_code)]
    fn view_hero() -> Html {
        html! {
            <section class="hero is-info is-bold is-large">
                <div class="hero-body">
                    <div class="container">
                        <h1 class="title">
                            { "Home" }
                        </h1>
                        <h2 class="subtitle">
                            { "症状图谱" }
                        </h2>
                    </div>
                </div>
            </section>
        }
    }

    fn view_chart(&self) -> Html {
        let node = html! {
                 <div class="block">
                     <div class="container network">
                         <div id="network" style="width:90%;height:1000px;margin:auto"></div>
                         <div id="stats_rels" style="width:90%;height:1000px;margin:auto"></div>
                     </div>
                 </div>
        };
        App::display_main();

        node
    }

    #[allow(dead_code)]
    fn view_info_tiles(&self) -> Html {
        html! {
            <>
                <div class="tile is-parent">
                    <div class="tile is-child box">
                        <p class="title">{ "What are yews?" }</p>
                        <p class="subtitle">{ "Everything you need to know!" }</p>

                        <div class="content">
                            {r#"
                            A yew is a small to medium-sized evergreen tree, growing 10 to 20 metres tall, with a trunk up to 2 metres in diameter.
                            The bark is thin, scaly brown, coming off in small flakes aligned with the stem.
                            The leaves are flat, dark green, 1 to 4 centimetres long and 2 to 3 millimetres broad, arranged spirally on the stem,
                            but with the leaf bases twisted to align the leaves in two flat rows either side of the stem,
                            except on erect leading shoots where the spiral arrangement is more obvious.
                            The leaves are poisonous.
                            "#}
                        </div>
                    </div>
                </div>

                <div class="tile is-parent">
                    <div class="tile is-child box">
                        <p class="title">{ "Who are we?" }</p>

                        <div class="content">
                            { "We're a small team of just 2" }
                            <sup>{ 64 }</sup>
                            { " members working tirelessly to bring you the low-effort yew content we all desperately crave." }
                            <br />
                            {r#"
                                We put a ton of effort into fact-checking our posts.
                                Some say they read like a Wikipedia article - what a compliment!
                            "#}
                        </div>
                    </div>
                </div>
            </>
        }
    }
}
