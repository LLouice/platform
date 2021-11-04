use anyhow::{anyhow, bail, Result};
use random;
use random::Source;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;
// use tensorflow::TensorType;
use tensorboard_rs::summary_writer::{FileWriter, SummaryWriter};
use tensorflow::ops::{self, create_summary_file_writer, summary_writer, write_summary};

#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn main() -> Result<()> {
    run()
}

fn run() -> Result<()> {
    let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let filename = dbg!(workspace_dir.parent().unwrap()).join("export/model.pb");
    dbg!(&filename);
    if !filename.exists() {
        bail!(format!(
            "Run 'python ada.py' to generate \
                     {} and try again.",
            filename.display()
        ));
    }

    // Generate some test data.
    /*
    let w = 0.1;
    let b = 0.3;
    let num_points = 100;
    let steps = 201;
    let mut rand = random::default();
    let mut x = Tensor::new(&[num_points as u64]);
    let mut y = Tensor::new(&[num_points as u64]);
    for i in 0..num_points {
        x[i] = (2.0 * rand.read::<f64>() - 1.0) as f32;
        y[i] = w * x[i] + b;
    }
    */

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;

    // session option
    let config_proto = include_bytes!("../assets/config.proto");
    // eprintln!("config_proto: {:?}", config_proto);
    let mut sess_opt = SessionOptions::new();
    sess_opt.set_config(config_proto.as_slice())?;
    let session = Session::new(&sess_opt, &graph)?;

    // get op from graph_def
    // the ops
    // init / op_optimize / op_file_path / op_save / op_summary

    let op_init = graph.operation_by_name_required("init")?;
    let op_batch_data_trn_init =
        graph.operation_by_name_required("data_trn/dataset_trn/MakeIterator")?;
    let op_batch_data_trn = graph.operation_by_name_required("data_trn/dataset_trn/get_next")?;

    let op_trn_record_path = graph.operation_by_name_required("data_trn/dataset_trn/Const")?;
    // based on current dir
    let trn_record_path: Tensor<String> = Tensor::from(String::from("assets/symptom_trn.tfrecord"));

    let op_optimize = graph.operation_by_name_required("optimizer/optimize")?;
    let op_global_step = graph.operation_by_name_required("optimizer/global_step")?;
    let op_loss = graph.operation_by_name_required("loss/loss")?;

    let op_file_path = graph.operation_by_name_required("save/Const")?;
    let op_save = graph.operation_by_name_required("save/control_dependency")?;
    let file_path_tensor: Tensor<String> = Tensor::from(String::from("checkpoints/saved.ckpt"));

    let op_summary = graph.operation_by_name_required("summaries/summary_op/summary_op")?;

    // Load the test data into the session.
    let mut init_step = SessionRunArgs::new();
    init_step.add_target(&op_init);
    init_step.add_feed(&op_trn_record_path, 0, &trn_record_path);
    init_step.add_target(&op_batch_data_trn_init);
    session.run(&mut init_step)?;

    // inspect the batch data
    let data_closure = || -> Result<()> {
        let mut inspect_data_step = SessionRunArgs::new();
        // inspect_data_step.add_target(&op_batch_data_trn_init);
        let batch_data_ix = inspect_data_step.request_fetch(&op_batch_data_trn, 0);

        for _ in 0..3 {
            session.run(&mut inspect_data_step)?;
            let batch_data: Tensor<i64> = inspect_data_step.fetch(batch_data_ix)?;
            // eprintln!("batch_data:\n\t{:?}", batch_data);
            eprintln!("batch_data:\n\t{}", batch_data);
        }
        Ok(())
    };
    // data_closure()?;

    // Train the model.
    let train_closure = || -> Result<()> {
        let epochs = 1;
        let mut train_step = SessionRunArgs::new();
        // train_step.add_feed(&op_x, 0, &x);
        // train_step.add_feed(&op_y, 0, &y);
        let loss_ix = train_step.request_fetch(&op_loss, 0);

        train_step.add_target(&op_optimize);
        for _ in 0..epochs {
            session.run(&mut train_step)?;
            let loss: Tensor<f64> = train_step.fetch(loss_ix)?;
            // eprintln!("batch_data:\n\t{:?}", batch_data);
            eprintln!("loss:\n\t{}", loss);
        }
        Ok(())
    };
    // train_closure()?;

    let mut writer = SummaryWriter::new(&("./logs/first".to_string()));

    let epochs = 10;
    let mut train_step = SessionRunArgs::new();
    let loss_token = train_step.request_fetch(&op_loss, 0);
    let global_step_token = train_step.request_fetch(&op_global_step, 0);
    // let summay_token = train_step.request_fetch(&op_summay, 0);
    train_step.add_target(&op_optimize);

    for _ in 0..epochs {
        session.run(&mut train_step)?;
        let loss: f32 = train_step.fetch(loss_token)?[0];
        eprintln!("loss:\n\t{}", loss);
        let global_step: i64 = train_step.fetch(global_step_token)?[0];
        // let summay: Tensor<String> = train_step.fetch(loss_token)?;
        writer.add_scalar("loss", loss, global_step as usize);
    }
    writer.flush();

    /*
    // Save the model.
    let mut step = SessionRunArgs::new();
    step.add_feed(&op_file_path, 0, &file_path_tensor);
    step.add_target(&op_save);
    session.run(&mut step)?;

    // Initialize variables, to erase trained data.
    session.run(&mut init_step)?;

    // Load the model.
    let op_load = graph.operation_by_name_required("save/restore_all")?;
    let mut step = SessionRunArgs::new();
    step.add_feed(&op_file_path, 0, &file_path_tensor);
    step.add_target(&op_load);
    session.run(&mut step)?;

    // Grab the data out of the session.
    let mut output_step = SessionRunArgs::new();
    let w_ix = output_step.request_fetch(&op_w, 0);
    let b_ix = output_step.request_fetch(&op_b, 0);
    session.run(&mut output_step)?;

    // Check our results.
    let w_hat: f32 = output_step.fetch(w_ix)?[0];
    let b_hat: f32 = output_step.fetch(b_ix)?[0];
    println!(
        "Checking w: expected {}, got {}. {}",
        w,
        w_hat,
        if (w - w_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    println!(
        "Checking b: expected {}, got {}. {}",
        b,
        b_hat,
        if (b - b_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    */
    Ok(())
}
