use anyhow::Result;
use serde::Serialize;
// use serde::Serialize;
use tensorflow::{
    Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY, PREDICT_INPUTS, PREDICT_OUTPUTS,
};

const SAVED_MODEL_DIR: &str = "SavedModel";

#[derive(Debug)]
pub struct AdaEInput(Vec<f32>);

// #[derive(Debug, Serialize)]
// pub struct AdaEPrediction {
//     pub label: u8,
//     pub confidence: f32,
// }

#[derive(Debug)]
pub struct AdaEModel {
    bundle: SavedModelBundle,
    input_e1_op: Operation,
    input_e1_index: i32,
    input_rel_op: Operation,
    input_rel_index: i32,
    output_op: Operation,
    output_index: i32,
}

impl AdaEModel {
    pub fn from_dir(export_dir: &str) -> Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, export_dir)?;

        let sig = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
        // let input_info = sig.get_input(PREDICT_INPUTS)?;
        let input_e1_info = sig.get_input("e1")?;
        let input_rel_info = sig.get_input("rel")?;
        let output_info = sig.get_output("prediciton")?;

        let input_e1_op = graph.operation_by_name_required(&input_e1_info.name().name)?;
        let input_rel_op = graph.operation_by_name_required(&input_rel_info.name().name)?;
        let output_op = graph.operation_by_name_required(&output_info.name().name)?;

        let input_e1_index = input_e1_info.name().index;
        let input_rel_index = input_rel_info.name().index;
        let output_index = output_info.name().index;

        Ok(Self {
            bundle,
            input_e1_op,
            input_e1_index,
            input_rel_op,
            input_rel_index,
            output_op,
            output_index,
        })
    }

    /*
    pub fn predict(&self, image: AdaEInput) -> Result<AdaEPrediction> {
        const INPUT_DIMS: &[u64] = &[1, 28, 28, 1];
        let input_tensor = Tensor::<f32>::new(INPUT_DIMS).with_values(&image.0)?;
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.input_op, self.input_index, &input_tensor);
        let output_fetch = run_args.request_fetch(&self.output_op, self.output_index);
        self.bundle.session.run(&mut run_args)?;

        let output = run_args.fetch::<f32>(output_fetch)?;
        let mut confidence = 0f32;
        let mut label = 0u8;
        for i in 0..output.dims()[1] {
            let conf = output[i as usize];
            if conf > confidence {
                confidence = conf;
                label = i as u8;
            }
        }

        Ok(AdaEPrediction { label, confidence })
    }
    */
}

mod tests {
    use super::*;

    #[test]
    fn smoke() {
        let model = AdaEModel::from_dir(SAVED_MODEL_DIR).expect("faield to load SavedModel");
        // eprintln!("model is: {model:?}");
    }
}
