#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::{json, Result, Value};

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
pub struct Category {
    name: String,
}

impl Category {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}
