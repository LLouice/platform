use crate::data::{NodeInfo, NodeLabel};
use anyhow::{anyhow, Result};
use serde::Serialize;
use serving::{AdaEInput, AdaEModel, AdaEPrediction};
use std::{collections::HashMap, fs::File, io::BufReader};

const NAME_ID_MAP: &str = "name_id_map.json";
const ID_NAME_MAP: &str = "id_name_map.json";

fn load_name_id_map() -> Result<HashMap<String, usize>> {
    let name_id_map: HashMap<String, usize> = {
        let reader = BufReader::new(File::open(NAME_ID_MAP)?);
        serde_json::from_reader(reader)?
    };
    println!("name_id_map len: {:?}", name_id_map.len());
    Ok(name_id_map)
}

fn load_id_name_map() -> Result<HashMap<usize, String>> {
    let id_name_map: HashMap<usize, String> = {
        let reader = BufReader::new(File::open(ID_NAME_MAP)?);
        serde_json::from_reader(reader)?
    };
    Ok(id_name_map)
}

// only support SymptomRelateXXX
impl NodeInfo {
    pub fn into_adae_input(&self, name_id_map: &HashMap<String, usize>) -> Result<AdaEInput> {
        let id = name_id_map
            .get(&self.to_string())
            .ok_or_else(|| anyhow!("name not in name_id_map"))?
            .to_owned() as i64; // ignore maybe overflow
        let e1 = vec![id; 5];
        let rel = vec![0, 1, 2, 3, 4];
        Ok(AdaEInput { e1, rel })
    }
}

#[derive(Debug, Serialize)]
pub struct AdaEPredictionWithName(pub Vec<Vec<(String, f32)>>);

impl AdaEPredictionWithName {
    pub fn new(prediction: AdaEPrediction, id_name_map: &HashMap<usize, String>) -> Self {
        AdaEPredictionWithName(
            prediction
                .0
                .into_iter()
                .map(|x| {
                    x.into_iter()
                        .map(|(idx, conf)| (id_name_map.get(&idx).unwrap().to_owned(), conf))
                        .collect::<Vec<(String, f32)>>()
                })
                .collect(),
        )
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_load_map() {
        let _ = load_name_id_map();
    }

    #[test]
    fn test_from_node_info() {
        let name_id_map = load_name_id_map().expect("failed to load map");
        let node_info = NodeInfo {
            label: NodeLabel::Symptom,
            name: "肩背痛".into(),
        };
        let adae_input = node_info.into_adae_input(&name_id_map).unwrap();
        eprintln!("{adae_input:?}")
    }

    #[test]
    fn test_predict() {
        let name_id_map = load_name_id_map().expect("failed to load map");
        let node_info = NodeInfo {
            label: NodeLabel::Symptom,
            name: "肩背痛".into(),
        };
        let adae_input = node_info.into_adae_input(&name_id_map).unwrap();
        // eprintln!("{adae_input:?}")

        let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let model_dir = workspace_dir
            .parent()
            .unwrap()
            .join("serving")
            .join(serving::SAVED_MODEL_DIR);

        dbg!(&model_dir);
        assert!(model_dir.is_dir());

        let model = AdaEModel::from_dir(model_dir.to_str().unwrap()).expect("failed to load model");
        let prediction = model.predict(adae_input).expect("failed to predict");
        dbg!(&prediction);
    }

    #[test]
    fn test_with_name() {
        let name_id_map = load_name_id_map().expect("failed to load map");
        let node_info = NodeInfo {
            label: NodeLabel::Symptom,
            name: "肩背痛".into(),
        };
        let adae_input = node_info.into_adae_input(&name_id_map).unwrap();
        // eprintln!("{adae_input:?}")

        let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let model_dir = workspace_dir
            .parent()
            .unwrap()
            .join("serving")
            .join(serving::SAVED_MODEL_DIR);

        // dbg!(&model_dir);
        assert!(model_dir.is_dir());

        let model = AdaEModel::from_dir(model_dir.to_str().unwrap()).expect("failed to load model");
        let prediction = model.predict(adae_input).expect("failed to predict");
        // dbg!(&prediction);

        // convert to AdaEPredictionWithName
        let id_name_map = load_id_name_map().expect("failed to load map");
        let prediction_with_name = AdaEPredictionWithName::new(prediction, &id_name_map);
        dbg!(&prediction_with_name);
    }
}
