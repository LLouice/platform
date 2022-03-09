use web_sys::HtmlElement;
use yew::prelude::*;

use crate::app::{App, NodeInfo};

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
        let data = r#"[[["Disease::颈椎退行性变,颈椎骨质增生",1.0],["Disease::偻附",0.9999993],["Disease::骨转移性恶性肿瘤",0.9999975],["Disease::全喉切除",0.9999149],["Disease::支气管肺癌",0.9998164],["Disease::巴特尔综",0.9995785],["Disease::背痈",0.9994726],["Disease::蛔虫病",0.99939597],["Disease::阴痹",0.99893606],["Disease::心脏供血不足",0.997449],["Disease::脊神经根炎",0.9965886],["Disease::小儿骨转移瘤",0.99060774],["Disease::胸椎病",0.9809451],["Disease::肩凝症,肩痹",0.84104645]],[["Drug::甲酚皂溶液",0.9999449],["Drug::过氧化氢",0.9996456],["Drug::驱风痛片(百会)",0.9959137],["Drug::快灵二合一(青松)",0.9954101],["Drug::胆石片",0.9430536],["Drug::根痛平片",0.9166226],["Drug::羌活胜湿汤",0.9126724],["Drug::大株红景天注射液",0.897667]],[["Department::中医科",1.0],["Department::骨科",0.99999976],["Department::骨科学",0.99999857],["Department::神经内科",0.9999824]],[["Check::骨关节及软组织CT",0.99998647]],[["Area::颅脑",0.73473996],["Area::全身",0.59496987]]]"#;
        let data: Vec<Vec<(String ,f32)>> = serde_json::from_str(&data).unwrap();
        let data: Vec<(String, f32)> = data.into_iter().flatten().collect();
        html! {
            <div class="columns is-centered mt-3">
            <table class="table is-bordered is-striped is-hoverable">
            <thead>
                <tr>
                <th><abbr title="Position">{"Pos"}</abbr></th>
                <th>{ "Name" }</th>
                <th>{"Confidence"}</th>
                </tr>
            </thead>
            <tfoot>
                <tr>
                <th><abbr title="Position">{"Pos"}</abbr></th>
                <th>{"Name"}</th>
                <th>{ "Confidence "}</th>
                </tr>
            </tfoot>
            <tbody>
                // <tr>
                // <th>{ "1" }</th>
                // <td>{ "A" }</td>
                // <td>{ "0.1" }</td>
                // </tr>
                // <tr>
                // <th>{ "2" }</th>
                // <td>{ "B" }</td>
                // <td>{ "0.9" }</td>
                // </tr>
                {

                    for data.into_iter().enumerate().map(|(i,(name, conf))| {
                        html!{
                            <tr>
                            <th>{ i+1 }</th>
                            <td>{ name }</td>
                            <td>{ conf }</td>
                            </tr>
                        }
                    }
                   )

                }

            </tbody>
            </table>
            </div>
        }
    }
}
