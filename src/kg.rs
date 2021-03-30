use crate::echarts::{Category, Link, Node as Enode};
use crate::neo4j::init_graph;
use chrono::NaiveDateTime;
use neo4rs::*;
use serde::Serialize;
use std::collections::HashMap;

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

pub type Nodes = Vec<Enode>;
pub type Links = Vec<Link>;
pub type Categories = Vec<Category>;

#[derive(Debug, Serialize)]
pub struct GraphData {
    data: Nodes,
    links: Links,
    categories: Categories,
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
        let main_node = Enode::new(0, name.to_owned(), des, 70, 0);
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
            let node = Enode::new(
                idx + 1,
                x.name.clone(),
                des,
                50,
                cats.get(&x.label)
                    .expect(format!("key {:?} not exists in cats", x.label).as_str())
                    .to_owned(),
            );
            nodes.push(node);

            // let link = Link::new(name.to_owned(), x.name.clone());
            let link = Link::new("0".to_owned(), format!("{}", idx));
            links.push(link);
        }
        Ok(GraphData {
            data: nodes,
            links,
            categories,
        })
    }
}

/*

def get_nodes(self, node_types=None, limit=-1):
assert node_types is not None, "please specify the node_types"
node_types = listify(node_types)
nodes = [[] for _ in node_types]

def _fn(cur, group):
nodes[group].append(dict(id=cur.current["p"]["name"], group=group))

if node_types:
for idx, node_type in enumerate(node_types):
_cypher = self.match_cypher.format(node_type=node_type)
if limit > 0:
_cypher += f" limit {limit}"
self.run_cypher(_cypher, cb=_fn, group=idx)
# 展开嵌套列表
all_nodes = flat(nodes)
if limit > 0:
# limit * len(node_types) -> limit
# 随机 sample
all_nodes = random.sample(all_nodes, limit)
return all_nodes
*/
