use crate::data::{NodeInfo, NodeLabel};
use anyhow::{anyhow, Result};
use serving::{AdaEInput, AdaEModel, AdaEPrediction};
use std::{collections::HashMap, fs::File, io::BufReader};

const NAME_ID_MAP: &str = "name_id_map.json";

fn load_map() -> Result<HashMap<String, usize>> {
    let name_id_map: HashMap<String, usize> = {
        let reader = BufReader::new(File::open(NAME_ID_MAP)?);
        serde_json::from_reader(reader)?
    };
    println!("name_id_map len: {:?}", name_id_map.len());
    Ok((name_id_map))
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

mod tests {
    use super::*;

    #[test]
    fn test_load_map() {
        let _ = load_map();
    }

    #[test]
    fn test_from_node_info() {
        let name_id_map = load_map().expect("failed to load map");
        let node_info = NodeInfo {
            label: NodeLabel::Symptom,
            name: "肩背痛".into(),
        };
        let adae_input = node_info.into_adae_input(&name_id_map).unwrap();
        eprintln!("{adae_input:?}")
    }

    #[test]
    fn test_predict() {
        let name_id_map = load_map().expect("failed to load map");
        let node_info = NodeInfo {
            label: NodeLabel::Symptom,
            name: "肩背痛".into(),
        };
        let adae_input = node_info.into_adae_input(&name_id_map).unwrap();
        // eprintln!("{adae_input:?}")

        let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let model_dir = workspace_dir
            .parent().unwrap()
            .join("serving")
            .join(serving::SAVED_MODEL_DIR);

        dbg!(&model_dir);
        assert!(model_dir.is_dir());

        let model = AdaEModel::from_dir(model_dir.to_str().unwrap()).expect("failed to load model");
        let prediction = model.predict(adae_input).expect("failed to predict");
        dbg!(&prediction);
    }
}
