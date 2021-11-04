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

const TRAIN_SIZE: i64 = 43743; // (h,r)
const VAL_SIZE: i64 = 21121;
const TEST_SIZE: i64 = 14038;

enum Dev {
    Val,
    Test,
}

struct EvalValue {
    rank: f32,
    hit1: f32,
    hit3: f32,
    hit10: f32,
}

impl EvalValue {
    fn new(rank: f32, hit1: f32, hit3: f32, hit10: f32) -> Self {
        Self {
            rank,
            hit1,
            hit3,
            hit10,
        }
    }
}

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

    // custom
    let batch_size_trn: i64 = 4;
    let batch_size_dev: i64 = 8;
    // resume from checkpoint
    let epochs: i64 = 1;
    let epoch_start: i64 = 1;
    let steps: i64 = TRAIN_SIZE / batch_size_trn;
    let repeat_num = epochs * steps;
    let batch_size_trn_tensor = Tensor::<i64>::new(&[]).with_values(&[batch_size_trn])?;
    let batch_size_dev_tensor = Tensor::<i64>::new(&[]).with_values(&[batch_size_dev])?;
    let repeat_num_tensor = Tensor::<i64>::new(&[]).with_values(&[repeat_num])?;
    let lr: f32 = 0.001;
    let lr_tensor = Tensor::<f32>::new(&[]).with_values(&[lr])?;

    // train stuff
    let eval_interval = 5;

    println!("{}", batch_size_trn_tensor);

    // get op from graph_def
    // the ops
    // init / placeholder / op_optimize / op_file_path / op_save / op_summary

    let op_init = graph.operation_by_name_required("init")?;
    let op_ph_batch_size_trn = graph.operation_by_name_required("custom/batch_size_trn")?;
    let op_ph_batch_size_dev = graph.operation_by_name_required("custom/batch_size_dev")?;
    let op_ph_repeat = graph.operation_by_name_required("custom/repeat")?;
    let op_ph_lr = graph.operation_by_name_required("custom/lr")?;

    // dataset
    let op_batch_data_trn_init =
        graph.operation_by_name_required("data_trn/dataset_trn/MakeIterator")?;
    let op_batch_data_trn = graph.operation_by_name_required("data_trn/dataset_trn/get_next")?;
    let op_trn_record_path = graph.operation_by_name_required("data_trn/dataset_trn/Const")?;
    // based on current dir
    let trn_record_path: Tensor<String> = Tensor::from(String::from("assets/symptom_trn.tfrecord"));

    let op_batch_data_val_init =
        graph.operation_by_name_required("data_val/dataset_val/MakeIterator")?;
    let op_batch_data_val = graph.operation_by_name_required("data_val/dataset_val/get_next")?;
    let op_val_record_path = graph.operation_by_name_required("data_val/dataset_val/Const")?;
    // based on current dir
    let val_record_path: Tensor<String> = Tensor::from(String::from("assets/symptom_val.tfrecord"));

    let op_batch_data_test_init =
        graph.operation_by_name_required("data_test/dataset_test/MakeIterator")?;
    let op_batch_data_test = graph.operation_by_name_required("data_test/dataset_test/get_next")?;
    let op_test_record_path = graph.operation_by_name_required("data_test/dataset_test/Const")?;
    // based on current dir
    let test_record_path: Tensor<String> =
        Tensor::from(String::from("assets/symptom_test.tfrecord"));

    // train op
    let op_optimize = graph.operation_by_name_required("optimizer/optimize")?;
    let op_global_step = graph.operation_by_name_required("optimizer/global_step")?;
    let op_loss = graph.operation_by_name_required("loss/loss")?;

    // eval op
    let op_rank_val = graph.operation_by_name_required("eval_val/rank_val")?;
    let op_hit1_val = graph.operation_by_name_required("eval_val/hits1_val")?;
    let op_hit3_val = graph.operation_by_name_required("eval_val/hits3_val")?;
    let op_hit10_val = graph.operation_by_name_required("eval_val/hits10_val")?;
    let op_eval_val = graph.operation_by_name_required("eval_val/eval_op_val")?;

    let op_rank_test = graph.operation_by_name_required("eval_test/rank_test")?;
    let op_hit1_test = graph.operation_by_name_required("eval_test/hits1_test")?;
    let op_hit3_test = graph.operation_by_name_required("eval_test/hits3_test")?;
    let op_hit10_test = graph.operation_by_name_required("eval_test/hits10_test")?;
    let op_eval_test = graph.operation_by_name_required("eval_test/eval_op_test")?;

    // save op
    let op_file_path = graph.operation_by_name_required("save/Const")?;
    let op_save = graph.operation_by_name_required("save/control_dependency")?;
    let file_path_tensor: Tensor<String> = Tensor::from(String::from("checkpoints/saved.ckpt"));

    // summary, not used here
    let op_summary = graph.operation_by_name_required("summaries/summary_op/summary_op")?;

    // Load the test data into the session.
    let mut init_step = SessionRunArgs::new();
    init_step.add_target(&op_init);
    init_step.add_feed(&op_ph_batch_size_trn, 0, &batch_size_trn_tensor);
    init_step.add_feed(&op_ph_batch_size_dev, 0, &batch_size_dev_tensor);
    init_step.add_feed(&op_ph_repeat, 0, &repeat_num_tensor);
    init_step.add_feed(&op_ph_lr, 0, &lr_tensor);

    init_step.add_feed(&op_trn_record_path, 0, &trn_record_path);
    init_step.add_target(&op_batch_data_trn_init);

    init_step.add_feed(&op_val_record_path, 0, &val_record_path);
    init_step.add_target(&op_batch_data_val_init);

    init_step.add_feed(&op_test_record_path, 0, &test_record_path);
    init_step.add_target(&op_batch_data_test_init);

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

    let save = |ckpt_name: &str| -> Result<()> {
        let file_path_tensor: Tensor<String> = Tensor::from(String::from(ckpt_name));
        let mut save_step = SessionRunArgs::new();
        save_step.add_feed(&op_file_path, 0, &file_path_tensor);
        save_step.add_target(&op_save);
        session.run(&mut save_step)?;
        Ok(())
    };

    let restore = |init_step: &mut SessionRunArgs, ckpt_name: &str| -> Result<()> {
        session.run(init_step)?;
        // Load the model.
        let op_load = graph.operation_by_name_required("save/restore_all")?;
        let file_path_tensor: Tensor<String> = Tensor::from(String::from(ckpt_name));
        let mut step = SessionRunArgs::new();
        step.add_feed(&op_file_path, 0, &file_path_tensor);
        step.add_target(&op_load);
        session.run(&mut step)?;
        Ok(())
    };

    // Train the model.
    let mut train = || -> Result<()> {
        let mut writer = SummaryWriter::new(&("./logs/first".to_string()));
        let mut train_step = SessionRunArgs::new();
        let loss_token = train_step.request_fetch(&op_loss, 0);
        let global_step_token = train_step.request_fetch(&op_global_step, 0);
        train_step.add_target(&op_optimize);

        let eval = |dev: Dev, writer: &mut SummaryWriter, global_step: i64| -> Result<EvalValue> {
            let mut eval_step = SessionRunArgs::new();
            eval_step.add_feed(&op_ph_batch_size_dev, 0, &batch_size_dev_tensor);

            let (rank_token, hit1_token, hit3_token, hit10_token) = match dev {
                Dev::Val => {
                    eval_step.add_target(&op_eval_val);
                    (
                        eval_step.request_fetch(&op_rank_val, 0),
                        eval_step.request_fetch(&op_hit1_val, 0),
                        eval_step.request_fetch(&op_hit3_val, 0),
                        eval_step.request_fetch(&op_hit10_val, 0),
                    )
                }
                Dev::Test => {
                    eval_step.add_target(&op_eval_test);
                    (
                        eval_step.request_fetch(&op_rank_test, 0),
                        eval_step.request_fetch(&op_hit1_test, 0),
                        eval_step.request_fetch(&op_hit3_test, 0),
                        eval_step.request_fetch(&op_hit10_test, 0),
                    )
                }
            };
            session.run(&mut eval_step)?;
            // fetch
            let rank: f32 = eval_step.fetch(rank_token)?[0];
            let hit1: f32 = eval_step.fetch(hit1_token)?[0];
            let hit3: f32 = eval_step.fetch(hit3_token)?[0];
            let hit10: f32 = eval_step.fetch(hit10_token)?[0];
            writer.add_scalar("rank", rank, global_step as usize);
            writer.add_scalar("hit1", hit1, global_step as usize);
            writer.add_scalar("hit3", hit3, global_step as usize);
            writer.add_scalar("hit10", hit10, global_step as usize);
            println!(
                "rank: {}, hit1: {} hit3: {}, hit10: {}",
                rank, hit1, hit3, hit10
            );
            writer.flush();
            let value = EvalValue::new(rank, hit1, hit3, hit10);
            Ok(value)
        };

        let mut global_step: i64 = 0;
        let steps = 100;
        // for e in 1..=epochs {
        for e in epoch_start..=5 {
            // for i in 0..steps {
            let mut total_loss: f32 = 0.;
            for i in 0..steps {
                session.run(&mut train_step)?;
                let loss: f32 = train_step.fetch(loss_token)?[0];
                total_loss += loss;
                // eprintln!("loss:\n\t{}", loss);
                global_step = train_step.fetch(global_step_token)?[0];
                writer.add_scalar("loss", loss, global_step as usize);
            }
            let loss = total_loss / steps as f32;
            println!("[{}] loss: {}", e, loss);
            if e % eval_interval == 0 {
                let eval_value = eval(Dev::Val, &mut writer, global_step)?;
                let checkpoint_name = format!(
                    "checkpoints/rank_{}_loss_{}_epoch_{}_step_{}",
                    eval_value.rank, loss, e, global_step
                );
                save(&checkpoint_name)?;
            }
        }
        writer.flush();
        Ok(())
    };
    train()?;

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
