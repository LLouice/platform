use chrono::NaiveDateTime;
use neo4rs::*;
use serde::Serialize;

use std::collections::{HashMap, HashSet};

use crate::data::{Category, GraphData, GraphDataD3, Link, LinkD3, Node as DNode};
use crate::neo4j::init_graph;

#[derive(Debug)]
pub struct KgResult {
    name: String,
    label: Vec<String>,
    updated_time: NaiveDateTime,
}

impl KgResult {
    fn new(name: String, label: Vec<String>, updated_time: NaiveDateTime) -> Self {
        KgResult {
            name,
            label,
            updated_time,
        }
    }
}

pub struct Kg;

impl Kg {
    pub async fn get_out_links(src_type: &str, name: &str) -> Vec<KgResult> {
        let graph = init_graph().await;
        let cypher = format!(
            "match (ps:{}{{name:$name}}) -[r]-> (pt) return ps,r,pt",
            src_type
        );
        let mut result = graph
            .execute(query(cypher.as_str()).param("name", name))
            .await
            .expect("graph execute error");

        let mut res = vec![];
        while let Ok(Some(row)) = result.next().await {
            let node: Node = row.get("pt").unwrap();
            let labels = node.labels();
            let name: String = node.get("name").unwrap();
            let updated_time: NaiveDateTime = node.get("updated_time").unwrap();
            res.push(KgResult::new(name, labels, updated_time));
        }
        res
    }

    pub fn convert(src_type: &str, name: &str, kg_res: Vec<KgResult>) -> anyhow::Result<GraphData> {
        let mut nodes = vec![];
        let mut links = vec![];
        let mut categories = vec![];

        let mut cats = HashMap::new();
        cats.insert(vec![src_type.to_owned()], 0);
        // categories.push(Category::new(src_type.to_owned()));
        categories.push(Category::new(src_type.to_owned()));

        let des = format!("{}::{}", src_type, name);
        let main_node = DNode::new(0, name.to_owned(), des, 70, 0);
        nodes.push(main_node);

        for x in kg_res.iter() {
            if !cats.contains_key(&x.label) {
                let id = cats.len();
                cats.insert(x.label.to_owned(), id);
                let label = x.label.join("-");
                categories.push(Category::new(label));
            }
        }
        println!("cats: {:?}", cats);

        for (idx, x) in kg_res.into_iter().enumerate() {
            let label = x.label.join("-");
            let des = format!("{}::{}", label, x.name);
            let node = DNode::new(
                idx + 1,
                x.name.clone(),
                des,
                50,
                cats.get(&x.label)
                    // .expect(format!("key {:?} not exists in cats", x.label).as_str())
                    .ok_or_else(|| {
                        let err_msg = format!("key {:?} not exists in cats", x.label);
                        anyhow!(err_msg)
                    })?
                    .to_owned(),
            );
            nodes.push(node);

            // let link = Link::new(name.to_owned(), x.);
            let link = Link::new("0".to_owned(), format!("{}", idx + 1));
            links.push(link);
        }
        Ok(GraphData {
            data: nodes,
            links,
            categories,
        })
    }

    pub fn convert_dedup(
        src_type: &str,
        name: &str,
        kg_res: Vec<KgResult>,
    ) -> anyhow::Result<GraphData> {
        let mut node_keys = HashSet::new();

        let mut nodes = vec![];
        let mut links = vec![];
        let mut categories = vec![];

        let mut cats = HashMap::new();
        cats.insert(vec![src_type.to_owned()], 0);
        // categories.push(Category::new(src_type.to_owned()));
        categories.push(Category::new(src_type.to_owned()));

        let des = format!("{}::{}", src_type, name);
        node_keys.insert(des.clone());
        let main_node = DNode::new(0, name.to_owned(), des, 70, 0);
        nodes.push(main_node);

        for x in kg_res.iter() {
            if !cats.contains_key(&x.label) {
                let id = cats.len();
                cats.insert(x.label.to_owned(), id);
                let label = x.label.join("-");
                categories.push(Category::new(label));
            }
        }
        println!("cats: {:?}", cats);

        for x in kg_res.into_iter() {
            let label = x.label.join("-");
            let des = format!("{}::{}", label, x.name);

            if !node_keys.contains(&des) {
                node_keys.insert(des.clone());

                let node_id = nodes.len();
                let node = DNode::new(
                    nodes.len(),
                    x.name.clone(),
                    des,
                    50,
                    cats.get(&x.label)
                        // .expect(format!("key {:?} not exists in cats", x.label).as_str())
                        .ok_or_else(|| {
                            let err_msg = format!("key {:?} not exists in cats", x.label);
                            anyhow!(err_msg)
                        })?
                        .to_owned(),
                );
                nodes.push(node);

                // let link = Link::new(name.to_owned(), x.);
                let link = Link::new("0".to_string(), node_id.to_string());
                links.push(link);
            }
        }

        Ok(GraphData {
            data: nodes,
            links,
            categories,
        })
    }

    pub fn convert_d3_dedup(
        src_type: &str,
        name: &str,
        kg_res: Vec<KgResult>,
    ) -> anyhow::Result<GraphDataD3> {
        let mut node_keys = HashSet::new();

        let mut nodes = vec![];
        let mut links = vec![];
        let mut categories = vec![];

        let mut cats = HashMap::new();
        cats.insert(vec![src_type.to_owned()], 0);
        // categories.push(Category::new(src_type.to_owned()));
        categories.push(Category::new(src_type.to_owned()));

        let des = format!("{}::{}", src_type, name);
        node_keys.insert(des.clone());
        let main_node = DNode::new(0, name.to_owned(), des, 70, 0);
        nodes.push(main_node);

        for x in kg_res.iter() {
            if !cats.contains_key(&x.label) {
                let id = cats.len();
                cats.insert(x.label.to_owned(), id);
                let label = x.label.join("-");
                categories.push(Category::new(label));
            }
        }
        println!("cats: {:?}", cats);

        for x in kg_res.into_iter() {
            let label = x.label.join("-");
            let des = format!("{}::{}", label, x.name);

            if !node_keys.contains(&des) {
                node_keys.insert(des.clone());

                let node_id = nodes.len();
                let node = DNode::new(
                    nodes.len(),
                    x.name.clone(),
                    des,
                    50,
                    cats.get(&x.label)
                        // .expect(format!("key {:?} not exists in cats", x.label).as_str())
                        .ok_or_else(|| {
                            let err_msg = format!("key {:?} not exists in cats", x.label);
                            anyhow!(err_msg)
                        })?
                        .to_owned(),
                );
                nodes.push(node);

                // let link = Link::new(name.to_owned(), x.);
                let link = LinkD3::new(0, node_id);
                links.push(link);
            }
        }

        Ok(GraphDataD3 {
            data: nodes,
            links,
            categories,
        })
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub type_name: String,
    pub count: usize,
    pub area: usize,
    pub check: usize,
    pub department: usize,
    pub drug: usize,
    pub disease: usize,
    pub symptom: usize,
}

impl Stats {
    pub fn new(type_name: String) -> Self {
        Stats {
            type_name,
            ..Default::default()
        }
    }
}

impl Kg {
    async fn node_num(graph: &Graph, src_type: &str) -> usize {
        let cypher = format!("match (ps:{}) return count(ps) as num", src_type);
        let mut result = graph
            .execute(query(cypher.as_str()))
            .await
            .expect("graph execute error");
        if let Ok(Some(row)) = result.next().await {
            return row.get::<i64>("num").unwrap() as usize;
        }
        0
    }

    async fn targ_num(graph: &Graph, src_type: &str, targ_type: &str) -> usize {
        let cypher = format!(
            "match (ps:{})-[r]-> (pt:{}) return count(pt) as num",
            src_type, targ_type
        );
        let mut result = graph
            .execute(query(cypher.as_str()))
            .await
            .expect("graph execute error");
        if let Ok(Some(row)) = result.next().await {
            return row.get::<i64>("num").unwrap() as usize;
        }
        0
    }

    pub async fn stat() -> Vec<Stats> {
        let graph = init_graph().await;
        // get node num
        let mut stats = vec![];
        for src_type in &["Symptom", "Disease", "Check"] {
            let mut s = Stats::new(src_type.to_string());

            let count = Self::node_num(&graph, src_type).await;
            println!("{}", count);
            s.count = count;

            for targ_type in &["Area", "Check", "Department", "Drug", "Disease", "Symptom"] {
                let count = Self::targ_num(&graph, src_type, targ_type).await;
                match *targ_type {
                    "Area" => s.area = count,
                    "Check" => s.check = count,
                    "Department" => s.department = count,
                    "Drug" => s.drug = count,
                    "Disease" => s.disease = count,
                    "Symptom" => s.symptom = count,
                    _ => unreachable!(),
                }
            }
            stats.push(s);
        }
        stats
    }

    pub async fn convert_stat(stats: Vec<Stats>) -> PieData {
        let mut nodes_pie = vec![];
        let mut rels_pie = vec![];
        let mut sym_pie = vec![];
        let mut dis_pie = vec![];
        let mut check_pie = vec![];

        // All nodes stat
        let graph = init_graph().await;
        for src_type in &["Symptom", "Disease", "Check", "Area", "Department", "Drug"] {
            let count = Self::node_num(&graph, src_type).await;
            nodes_pie.push(Pie::new(src_type, count as f64));
        }

        // sym dis check Stat and aggr all rels
        for s in stats {
            let name = s.type_name;
            // nodes_pie.push(Pie::new(name, s.count as f64));
            let the_pie = match name.as_str() {
                "Symptom" => &mut sym_pie,
                "Disease" => &mut dis_pie,
                "Check" => &mut check_pie,
                _ => unreachable!(),
            };
            if s.area > 0 {
                the_pie.push(Pie::new("Area", s.area as f64));
                rels_pie.push(Pie::new(format!("{} > {}", name, "Area"), s.area as f64));
            }

            if s.check > 0 {
                the_pie.push(Pie::new("Check", s.check as f64));
                rels_pie.push(Pie::new(format!("{} > {}", name, "Check"), s.check as f64));
            }
            if s.department > 0 {
                the_pie.push(Pie::new("Department", s.department as f64));
                rels_pie.push(Pie::new(
                    format!("{} > {}", name, "Department"),
                    s.department as f64,
                ));
            }
            if s.drug > 0 {
                the_pie.push(Pie::new("Drug", s.drug as f64));
                rels_pie.push(Pie::new(format!("{} > {}", name, "Drug"), s.drug as f64));
            }
            if s.disease > 0 {
                the_pie.push(Pie::new("Disease", s.disease as f64));
                rels_pie.push(Pie::new(
                    format!("{} > {}", name, "Disease"),
                    s.disease as f64,
                ));
            }
            if s.symptom > 0 {
                the_pie.push(Pie::new("Symptom", s.symptom as f64));
                rels_pie.push(Pie::new(
                    format!("{} > {}", name, "Symptom"),
                    s.symptom as f64,
                ));
            }
        }
        PieData {
            nodes_pie,
            rels_pie,
            sym_pie,
            dis_pie,
            check_pie,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Pie {
    name: String,
    value: f64,
}

impl Pie {
    fn new<S: ToString>(name: S, value: f64) -> Self {
        let name = name.to_string();
        Pie { name, value }
    }
}

pub type Pies = Vec<Pie>;

#[derive(Debug, Serialize)]
pub struct PieData {
    nodes_pie: Pies,
    rels_pie: Pies,
    sym_pie: Pies,
    dis_pie: Pies,
    check_pie: Pies,
}
