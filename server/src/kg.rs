use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use chrono::NaiveDateTime;
use neo4rs::*;
use serde::{Deserialize, Serialize};

use crate::data::{Category, GraphData, GraphDataD3, Link, LinkD3, Node as DNode, NodeInfo, NodeLabel, QRandomSample};
use crate::neo4j::init_graph;
use crate::session::GraphSession;

#[derive(Debug, Serialize)]
pub struct LinksResult {
    name: String,
    label: Vec<String>,
    #[serde(skip)]
    updated_time: NaiveDateTime,
}

impl LinksResult {
    fn new(name: String, label: Vec<String>, updated_time: NaiveDateTime) -> Self {
        LinksResult {
            name,
            label,
            updated_time,
        }
    }
}

pub struct Kg;

impl Kg {
    // FIXME: eliminate the unwrap and expect
    pub async fn get_out_links(node_label: NodeLabel, name: &str) -> Vec<LinksResult> {
        let graph = init_graph().await;
        let cypher = format!(
            "match (ps:{}{{name:$name}}) -[r]-> (pt) return ps,r,pt",
            node_label
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
            res.push(LinksResult::new(name, labels, updated_time));
        }
        res
    }

    pub async fn get_in_links(node_label: NodeLabel, name: &str) -> Vec<LinksResult> {
        let graph = init_graph().await;
        let cypher = format!(
            "match (ps) -[r]-> (pt:{}{{name:$name}})  return ps,r,pt",
            node_label
        );
        let mut result = graph
            .execute(query(cypher.as_str()).param("name", name))
            .await
            .expect("graph execute error");

        let mut res = vec![];
        while let Ok(Some(row)) = result.next().await {
            let node: Node = row.get("ps").unwrap();
            let labels = node.labels();
            let name: String = node.get("name").unwrap();
            let updated_time: NaiveDateTime = node.get("updated_time").unwrap();
            res.push(LinksResult::new(name, labels, updated_time));
        }
        res
    }

    #[deprecated(note = "Use `convert_dedup` instead.")]
    pub fn convert(src_type: &str, name: &str, kg_res: Vec<LinksResult>) -> anyhow::Result<GraphData> {
        let mut nodes = vec![];
        let mut links = vec![];
        let mut categories = vec![];

        let mut cats = HashMap::new();
        cats.insert(src_type.to_owned(), 0);
        // categories.push(Category::new(src_type.to_owned()));
        categories.push(Category::new(src_type.to_owned()));

        let des = format!("{}::{}", src_type, name);
        let main_node = DNode::new(0, name.to_owned(), des, 70, 0);
        nodes.push(main_node);

        for x in kg_res.iter() {
            let label = x.label.join("-");
            if !cats.contains_key(label.as_str()) {
                let id = cats.len();
                cats.insert(label.clone(), id);
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
                cats.get(label.as_str())
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
        label: &NodeLabel,
        name: &str,
        kg_res: Vec<LinksResult>,
    ) -> anyhow::Result<(GraphData, GraphSession)> {
        let mut node_keys = HashMap::new();

        let mut nodes = vec![];
        let mut links = vec![];
        let mut categories = vec![];

        let mut cats: HashMap<String, usize> = HashMap::new();
        cats.insert(label.to_string(), 0);
        categories.push(Category::new(label.to_string()));

        let des = format!("{}::{}", label, name);
        let node_id = node_keys.len();
        let _ = *node_keys.entry(des.clone()).or_insert(node_id);

        let main_node = DNode::new(0, name.to_owned(), des, 70, 0);
        nodes.push(main_node);

        for x in kg_res.iter() {
            let label = x.label.join("-");
            if !cats.contains_key(label.as_str()) {
                let id = cats.len();
                cats.insert(label.clone(), id);
                categories.push(Category::new(label));
            }
        }
        println!("cats: {:?}", cats);

        for x in kg_res.into_iter() {
            let label = x.label.join("-");
            let des = format!("{}::{}", label, x.name);

            if !node_keys.contains_key(&des) {
                let node_id = node_keys.len();
                node_keys.insert(des.clone(), node_id);

                let node = DNode::new(
                    node_id,
                    x.name.clone(),
                    des,
                    50,
                    cats.get(label.as_str())
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

        let graph_data = GraphData {
            data: nodes,
            links,
            categories,
        };
        let graph_session = GraphSession { node_keys, cats };

        Ok((graph_data, graph_session))
    }

    /// FIXME: merge with no session function
    /// no dedup in global context
    /// session for increase node and links
    /// current frontend state for readd into deleted node and link: in sess but not in current
    /// state, readd it
    pub fn convert_dedup_with_session(
        node_info: &NodeInfo,
        current_nodes_id: HashSet<usize>,
        current_links_set: HashSet<(usize, usize)>,
        kg_res: Vec<LinksResult>,
        sess: GraphSession,
    ) -> anyhow::Result<(GraphData, GraphSession)> {
        let NodeInfo { label, name } = node_info;

        // this is sess is a new copy Deserialized from the inner String
        let GraphSession {
            mut node_keys,
            mut cats,
        } = sess;

        // get current node info
        let current_node_des = format!("{}::{}", label, name);
        let current_node_id = node_keys.get(&current_node_des).unwrap().to_owned();

        let mut nodes = vec![];
        let mut links = vec![];
        // FIXME: contains current categories
        let mut categories = vec![];

        // current main node
        // let node_id = node_keys.len();
        // *node_keys.entry(des.clone()).or_insert(node_id);

        // FIXME: frontend give all categories and filter
        for x in kg_res.iter() {
            let label = x.label.join("-");
            if !cats.contains_key(label.as_str()) {
                let id = cats.len();
                cats.insert(label.clone(), id);
                categories.push(Category::new(label));
            }
        }

        println!("cats: {:?}", cats);

        // HashMap filter dup node
        for x in kg_res.into_iter() {
            // left node info
            let label = x.label.join("-");
            let des = format!("{}::{}", label, x.name);

            let is_node_contained = node_keys.contains_key(&des);

            if !is_node_contained {
                // new node, add node, add link
                let node_id = node_keys.len();
                node_keys.insert(des.clone(), node_id);

                let node = DNode::new(
                    node_id,
                    x.name.clone(),
                    des,
                    50,
                    cats.get(label.as_str())
                        // .expect(format!("key {:?} not exists in cats", x.label).as_str())
                        .ok_or_else(|| {
                            let err_msg = format!("key {:?} not exists in cats", x.label);
                            anyhow!(err_msg)
                        })?
                        .to_owned(),
                );
                nodes.push(node);

                let link = Link::new(current_node_id.to_string(), node_id.to_string());
                links.push(link);
            } else {
                // old node, just add to graph_data
                let node_id = node_keys.get(&des).unwrap().to_owned();

                // link is not presented
                let is_node_presented = current_nodes_id.contains(&node_id);
                let is_link_presented = current_links_set.contains(&(current_node_id, node_id));

                match (is_node_presented, is_link_presented) {
                    (false, _) => {
                        // add node, add link
                        let node = DNode::new(
                            node_id,
                            x.name.clone(),
                            des,
                            50,
                            cats.get(label.as_str())
                                // .expect(format!("key {:?} not exists in cats", x.label).as_str())
                                .ok_or_else(|| {
                                    let err_msg = format!("key {:?} not exists in cats", x.label);
                                    anyhow!(err_msg)
                                })?
                                .to_owned(),
                        );
                        nodes.push(node);
                        let link = Link::new(current_node_id.to_string(), node_id.to_string());
                        links.push(link);
                    }
                    (true, false) => {
                        // add link
                        let link = Link::new(current_node_id.to_string(), node_id.to_string());
                        links.push(link);
                    }
                    (true, true) => {}
                }
            }
        }

        let graph_data = GraphData {
            data: nodes,
            links,
            categories,
        };
        let graph_session = GraphSession { node_keys, cats };

        Ok((graph_data, graph_session))
    }

    pub fn convert_d3_dedup(
        node_label: NodeLabel,
        name: &str,
        kg_res: Vec<LinksResult>,
    ) -> anyhow::Result<GraphDataD3> {
        let mut node_keys = HashSet::new();

        let mut nodes = vec![];
        let mut links = vec![];
        let mut categories = vec![];

        let mut cats = HashMap::new();
        cats.insert(vec![node_label.to_string()], 0);
        categories.push(Category::new(node_label.to_string()));

        let des = format!("{}::{}", node_label, name);
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

    pub async fn query_node_links(NodeInfo { label, name }: &NodeInfo) -> Vec<LinksResult> {
        if *label == NodeLabel::Symptom {
            Self::get_out_links(*label, name.as_str()).await
        } else {
            Self::get_in_links(*label, name.as_str()).await
        }
    }

    pub async fn random_sample(QRandomSample { label, limit }: QRandomSample) -> Result<Vec<RandomSampleResult>> {
        let limit = limit.unwrap_or(10);

        let graph = init_graph().await;
        let cypher_out = format!(
            "MATCH(node:{}) -[r]-> (t) WHERE rand() < 0.5 RETURN node, COUNT(DISTINCT t) as num LIMIT {}",
            label,
            limit
        );
        let cypher_in = format!(
            "MATCH(h) -[r]-> (node:{}) WHERE rand() < 0.5 RETURN node, COUNT(DISTINCT h) as num LIMIT {}",
            label,
            limit
        );
        let cypher = match label {
            NodeLabel::Symptom => cypher_out,
            _ => cypher_in,
        };
        log::debug!("random_sample: {:?}", cypher);
        let mut result = graph
            .execute(query(cypher.as_str()))
            .await.map_err(|_| anyhow!("error on execute cypher"))?;

        // get query result
        let mut res = vec![];
        while let Ok(Some(row)) = result.next().await {
            log::debug!("row is: {:#?}\n\n", row);

            let node: Node = row.get("node").ok_or_else(|| anyhow!("no key `node` in row"))?;
            let name: String = node.get("name").ok_or_else(|| anyhow!("no key `name` in node"))?;
            let num = row.get::<i64>("num").ok_or_else(|| anyhow!("no key `num` in row"))?;
            // redundant info for future usage
            let updated_time: NaiveDateTime = node.get("updated_time").ok_or_else(|| anyhow!("no key `updated_time` in node"))?;

            res.push(RandomSampleResult { name, num, updated_time });
        }
        Ok(res)
    }
}


#[derive(Debug, Serialize)]
pub struct RandomSampleResult {
    name: String,
    #[serde(rename = "value")]
    num: i64,
    #[serde(skip)]
    updated_time: NaiveDateTime,
}


#[derive(Default, Debug)]
pub struct Stats {
    pub label: NodeLabel,
    pub count: usize,
    pub area: usize,
    pub check: usize,
    pub department: usize,
    pub drug: usize,
    pub disease: usize,
    pub symptom: usize,
}

impl Stats {
    pub fn new(label: NodeLabel) -> Self {
        Stats {
            label,
            ..Default::default()
        }
    }
}

impl Kg {
    async fn node_num(graph: &Graph, node_label: NodeLabel) -> usize {
        let cypher = format!("match (ps:{}) return count(ps) as num", node_label);
        let mut result = graph
            .execute(query(cypher.as_str()))
            .await
            .expect("graph execute error");
        if let Ok(Some(row)) = result.next().await {
            return row.get::<i64>("num").unwrap() as usize;
        }
        0
    }

    async fn targ_num(graph: &Graph, src_label: NodeLabel, targ_label: NodeLabel) -> usize {
        let cypher = format!(
            "match (ps:{})-[r]-> (pt:{}) return count(pt) as num",
            src_label, targ_label
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
        for src_label in [NodeLabel::Symptom, NodeLabel::Disease, NodeLabel::Check] {
            let mut s = Stats::new(src_label);

            let count = Self::node_num(&graph, src_label).await;
            println!("{}", count);
            s.count = count;

            for targ_label in [NodeLabel::Area, NodeLabel::Check, NodeLabel::Department, NodeLabel::Drug, NodeLabel::Disease, NodeLabel::Symptom] {
                let count = Self::targ_num(&graph, src_label, targ_label).await;
                match targ_label {
                    NodeLabel::Area => s.area = count,
                    NodeLabel::Check => s.check = count,
                    NodeLabel::Department => s.department = count,
                    NodeLabel::Drug => s.drug = count,
                    NodeLabel::Disease => s.disease = count,
                    NodeLabel::Symptom => s.symptom = count,
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

        // for src_type in &["Symptom", "Disease", "Check", "Area", "Department", "Drug"] {
        for src_label in [NodeLabel::Area, NodeLabel::Check, NodeLabel::Department, NodeLabel::Drug, NodeLabel::Disease, NodeLabel::Symptom] {
            let count = Self::node_num(&graph, src_label).await;
            nodes_pie.push(Pie::new(src_label, count as f64));
        }

        // sym dis check Stat and aggr all rels
        for s in stats {
            let node_label = s.label;
            // nodes_pie.push(Pie::new(name, s.count as f64));
            let the_pie = match node_label {
                NodeLabel::Symptom => &mut sym_pie,
                NodeLabel::Disease => &mut dis_pie,
                NodeLabel::Check => &mut check_pie,
                _ => unreachable!(),
            };
            if s.area > 0 {
                the_pie.push(Pie::new("Area", s.area as f64));
                rels_pie.push(Pie::new(format!("{} > {}", node_label, "Area"), s.area as f64));
            }

            if s.check > 0 {
                the_pie.push(Pie::new("Check", s.check as f64));
                rels_pie.push(Pie::new(format!("{} > {}", node_label, "Check"), s.check as f64));
            }
            if s.department > 0 {
                the_pie.push(Pie::new("Department", s.department as f64));
                rels_pie.push(Pie::new(
                    format!("{} > {}", node_label, "Department"),
                    s.department as f64,
                ));
            }
            if s.drug > 0 {
                the_pie.push(Pie::new("Drug", s.drug as f64));
                rels_pie.push(Pie::new(format!("{} > {}", node_label, "Drug"), s.drug as f64));
            }
            if s.disease > 0 {
                the_pie.push(Pie::new("Disease", s.disease as f64));
                rels_pie.push(Pie::new(
                    format!("{} > {}", node_label, "Disease"),
                    s.disease as f64,
                ));
            }
            if s.symptom > 0 {
                the_pie.push(Pie::new("Symptom", s.symptom as f64));
                rels_pie.push(Pie::new(
                    format!("{} > {}", node_label, "Symptom"),
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

///////////////////////////
// api query / payload info
///////////////////////////
#[derive(Debug, Deserialize)]
pub struct IncreaseUpdateState {
    pub node_info: NodeInfo,
    pub current_nodes_id: Vec<usize>,
    pub current_links_pair: Vec<(usize, usize)>,
}


#[cfg(test)]
mod tests {
    use super::*;

// use crate::utils;

    #[test]
    fn random_sample() {
        crate::init_env_logger!();
        let rt = tokio::runtime::Runtime::new().expect("get tokio Runtime fail!");


        rt.block_on(
            async {
                let query = QRandomSample::default();
                let res = Kg::random_sample(query).await;
                // debug_assert!(res.is_ok());
                // eprintln!("{:?}", res.unwrap());
                eprintln!("{:?}", res);
            }
        );
        log::debug!("end of test");
    }
}
