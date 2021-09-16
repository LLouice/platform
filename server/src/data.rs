#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub name: String,
    #[serde(alias = "describle")]
    pub des: String,
    #[serde(alias = "symbolSize", rename = "symbolSize")]
    pub symbol_size: usize,
    #[serde(skip, flatten)]
    x: Option<f32>,
    #[serde(skip, flatten)]
    y: Option<f32>,
    #[serde(skip, flatten)]
    value: Option<f32>,
    category: usize,
}

impl Node {
    pub fn new(id: usize, name: String, des: String, symbol_size: usize, category: usize) -> Self {
        let (x, y, value) = (None, None, None);
        Node {
            id,
            name,
            des,
            symbol_size,
            x,
            y,
            value,
            category,
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.des == other.des
    }
}
impl Eq for Node {}

// ------ Link ------
#[derive(Debug, Serialize, Deserialize)]
pub struct Link {
    source: String,
    target: String,
    #[serde(skip, flatten)]
    name: Option<String>,
    #[serde(skip, flatten)]
    des: Option<String>,
}

impl Link {
    pub fn new(source: String, target: String) -> Self {
        let (name, des) = (None, None);
        Link {
            source,
            target,
            name,
            des,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LinkD3 {
    source: usize,
    target: usize,
    #[serde(skip, flatten)]
    name: Option<String>,
    #[serde(skip, flatten)]
    des: Option<String>,
}

impl LinkD3 {
    pub fn new(source: usize, target: usize) -> Self {
        let (name, des) = (None, None);
        LinkD3 {
            source,
            target,
            name,
            des,
        }
    }
}

impl PartialEq for LinkD3 {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.target == self.target
    }
}
impl Eq for LinkD3 {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Category {
    name: String,
}

impl Category {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

pub type Nodes = Vec<Node>;
pub type LinksD3 = Vec<LinkD3>;
pub type Links = Vec<Link>;
pub type Categories = Vec<Category>;

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub data: Nodes,
    pub links: Links,
    pub categories: Categories,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphDataD3 {
    pub data: Nodes,
    pub links: LinksD3,
    pub categories: Categories,
}
