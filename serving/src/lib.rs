use anyhow::Result;
use serde::Serialize;
// use serde::Serialize;
use tensorflow::{
    Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY, PREDICT_INPUTS, PREDICT_OUTPUTS,
};

pub const SAVED_MODEL_DIR: &str = "SavedModel";
pub const THRESHOLD: f32 = 0.5;

#[derive(Debug)]
pub struct AdaEInput {
    pub e1: Vec<i64>,
    pub rel: Vec<i64>,
}

#[derive(Debug, Serialize)]
pub struct AdaEPrediction(pub Vec<Vec<(usize, f32)>>);

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

    // now the length is fixed at 5
    pub fn predict(&self, input: AdaEInput) -> Result<AdaEPrediction> {
        let AdaEInput { e1, rel } = input;
        let input_e1_tensor = Tensor::<i64>::new(&[e1.len() as u64]).with_values(e1.as_slice())?;
        let input_rel_tensor =
            Tensor::<i64>::new(&[rel.len() as u64]).with_values(rel.as_slice())?;

        let mut run_args = SessionRunArgs::new();

        run_args.add_feed(&self.input_e1_op, self.input_e1_index, &input_e1_tensor);
        run_args.add_feed(&self.input_rel_op, self.input_rel_index, &input_rel_tensor);
        let output_fetch = run_args.request_fetch(&self.output_op, self.output_index);
        self.bundle.session.run(&mut run_args)?;

        let output = run_args.fetch::<f32>(output_fetch)?;
        let dims = output.dims();
        let i = dims[0];
        let j = dims[1];

        let mut prediction = Vec::with_capacity(output.dims()[0] as usize);
        for ii in 0..i {
            let mut row = Vec::new();
            for jj in 0..j {
                let conf = output.get(&[ii, jj]);
                if conf > THRESHOLD {
                    row.push((jj as usize, conf));
                }
            }
            prediction.push(row);
        }
        Ok(AdaEPrediction(prediction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        let model = AdaEModel::from_dir(SAVED_MODEL_DIR).expect("faield to load SavedModel");
        // eprintln!("model is: {model:?}");
    }
}
