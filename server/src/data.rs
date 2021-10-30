#![allow(dead_code)]

use std::fmt::{Display, Formatter};

use anyhow;
use serde::{Deserialize, Serialize};
use strum::EnumIter;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub name: String,
    #[serde(alias = "describe")]
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


#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumIter)]
pub enum NodeLabel {
    Symptom,
    Disease,
    Drug,
    Department,
    Check,
    Area,
}

impl Default for NodeLabel {
    fn default() -> Self {
        NodeLabel::Symptom
    }
}

impl std::fmt::Display for NodeLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cat = match self {
            NodeLabel::Symptom => "Symptom",
            NodeLabel::Disease => "Disease",
            NodeLabel::Drug => "Drug",
            NodeLabel::Department => "Department",
            NodeLabel::Check => "Check",
            NodeLabel::Area => "Area",
            _ => "Unknown",
        };
        write!(f, "{}", cat)
    }
}

impl core::str::FromStr for NodeLabel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Symptom" => Ok(NodeLabel::Symptom),
            "Disease" => Ok(NodeLabel::Disease),
            "Drug" => Ok(NodeLabel::Drug),
            "Department" => Ok(NodeLabel::Department),
            "Check" => Ok(NodeLabel::Check),
            "Area" => Ok(NodeLabel::Area),
            _ => Err(anyhow::anyhow!("ParseNodeLabelError")),
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QRandomSample {
    pub label: NodeLabel,
    pub limit: Option<usize>,
}

impl Display for QRandomSample {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let limit = self.limit.unwrap_or(10);
        write!(f, "{}&limit={}", self.label, limit)
    }
}

impl core::str::FromStr for QRandomSample {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let stuff: Vec<&str> = s.trim()
            .split("&limit=")
            .collect();
        let label = stuff[0].parse::<NodeLabel>();
        let limit = stuff[1].parse::<usize>();

        match (label, limit) {
            (Ok(label), Ok(limit)) => Ok(QRandomSample { label, limit: Some(limit) }),
            _ => Err(anyhow::anyhow!("ParseQRandomSampleError")),
        }
    }
}

///////////////////////////
// api query / payload info
///////////////////////////
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NodeInfo {
    pub label: NodeLabel,
    pub name: String,
}


impl Display for NodeInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", self.label, self.name)
    }
}

impl core::str::FromStr for NodeInfo {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let stuff: Vec<&str> = s.trim()
            .split("::")
            .collect();
        let label = stuff[0].parse::<NodeLabel>();
        let name = urlencoding::decode(stuff[1]).map(|x| x.into_owned());
        match (label, name) {
            (Ok(label), Ok(name)) => Ok(Self { label, name }),
            _ => Err(anyhow::anyhow!("ParseNodeInfoError")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urldecode() {
        let raw_str = "%E4%BF%A1%E5%8F%AF%E6%AD%A2";
        let res = urlencoding::decode(raw_str);
        eprintln!("\nres: {:?}\n", res);
        assert_eq!(res.unwrap().into_owned(), "信可止");
    }

    #[test]
    fn urlencode() {
        let raw_str = "信可止";
        let res = urlencoding::encode(raw_str);
        eprintln!("\nres: {:?}\n", res);
    }
}
