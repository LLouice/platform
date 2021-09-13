use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// FIXME: rename it to nodes_map
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphSession {
    pub node_keys: HashMap<String, usize>,
    /// cats: labels combined identifies and it's index
    pub cats: HashMap<String, usize>,
}
