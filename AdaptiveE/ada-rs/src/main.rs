use std::convert::TryFrom;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::str::FromStr;

use anyhow::{anyhow, bail, Result};
use clap::{App, Arg, ArgMatches};
use random;
use random::Source;
// use tensorflow::TensorType;
use tensorboard_rs::summary_writer::{FileWriter, SummaryWriter};
use tensorflow::ops::{self, create_summary_file_writer, summary_writer, write_summary};
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

use ada_rs::init_env_logger;
use ada_rs::proto::build_config_proto;

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

#[derive(Debug)]
struct TrainConfig {
    visible_device_list: String,
    log_device_placement: bool,
    batch_size_trn: i64,
    batch_size_dev: i64,
    lr: f32,
    epochs: i64,
    steps: Option<i64>,
    eval_interval: i64,
    ckpt_dir: String,
    ckpt: Option<String>,
    ex: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            visible_device_list: String::from("2"),
            log_device_placement: false,
            batch_size_trn: 4,
            batch_size_dev: 8,
            lr: 0.001,
            epochs: 100,
            eval_interval: 5,
            ckpt_dir: "checkpoints".to_string(),
            ..Default::default()
        }
    }
}

// section TrainConfig-parse
impl TryFrom<&ArgMatches<'_>> for TrainConfig {
    type Error = anyhow::Error;

    fn try_from(args: &ArgMatches) -> Result<Self> {
        let steps = match args.value_of("steps") {
            Some(x) => Some(x.parse::<i64>()?),
            None => None,
        };

        Ok(Self {
            visible_device_list: args.value_of("visible_device_list").unwrap().to_owned(),
            log_device_placement: args.is_present("log_device_placement"),
            batch_size_trn: args
                .value_of("batch_size_trn")
                .map_or_else(|| Ok(4), |x| x.parse::<i64>())?,
            batch_size_dev: args
                .value_of("batch_size_dev")
                .map_or_else(|| Ok(8), |x| x.parse::<i64>())?,
            lr: args
                .value_of("lr")
                .map_or_else(|| Ok(0.001), |x| x.parse::<f32>())?,
            epochs: args
                .value_of("epochs")
                .map_or_else(|| Ok(100), |x| x.parse::<i64>())?,
            steps: steps,
            eval_interval: args
                .value_of("eval_interval")
                .map_or_else(|| Ok(5), |x| x.parse::<i64>())?,
            ckpt_dir: args.value_of("ckpt_dir").unwrap().to_owned(),
            ckpt: args.value_of("ckpt").map(str::to_owned),
            ex: args.value_of("ex").unwrap().to_owned(),
        })
    }
}

#[derive(Debug)]
struct Ckpt {
    rank: f32,
    loss: f32,
    epoch: i64,
    step: i64,
}

impl Ckpt {
    fn new(rank: f32, loss: f32, epoch: i64, step: i64) -> Self {
        Ckpt {
            rank,
            loss,
            epoch,
            step,
        }
    }
}

impl Display for Ckpt {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rank_{}_loss_{}_epoch_{}_step_{}",
            self.rank, self.loss, self.epoch, self.step
        )
    }
}

impl FromStr for Ckpt {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let split: Vec<&str> = s[..s.rfind('.').unwrap_or(s.len())]
            .trim()
            .split("_")
            .enumerate()
            .filter_map(|(i, x)| if i & 1 == 1 { Some(x) } else { None })
            .collect();
        dbg!(&split);
        let rank = split[0].parse::<f32>()?;
        let loss = split[1].parse::<f32>()?;
        let epoch = split[2].parse::<i64>()?;
        let step = split[3].parse::<i64>()?;
        Ok(Self {
            rank,
            loss,
            epoch,
            step,
        })
    }
}

fn main() -> Result<()> {
    init_env_logger!();
    let mut app = App::new("AdaE-train")
        .version("1.0")
        .author("LLouice")
        .about("train AdaE with tensorlfow")
        // .license("MIT OR Apache-2.0")
        .arg(
            Arg::with_name("visible_device_list")
                .short("v")
                .long("visible_device_list")
                .default_value("2")
                .help("set visible devices, like `2,3`")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("log_device_placement")
                .short("V")
                .long("log_device_placement")
                .help("Whether device placements should be logged"),
        )
        .arg(
            Arg::with_name("batch_size_trn")
                .long("bs_trn")
                .default_value("4")
                .help("batch size trn")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("batch_size_dev")
                .long("bs_dev")
                .default_value("8")
                .help("batch size dev")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("lr")
                .long("lr")
                .default_value("0.001")
                .help("learning rate")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("epochs")
                .short("e")
                .long("epochs")
                .default_value("100")
                .help("train epochs")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("steps")
                .long("steps")
                .help("steps")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("eval_interval")
                .short("I")
                .long("eval_interval")
                .default_value("5")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ckpt_dir")
                .short("D")
                .long("ckpt_dir")
                .help("checkpoint dir")
                .default_value("checkpoints")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ckpt")
                .short("R")
                .long("ckpt")
                .help("checkpoint for resume training")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ex")
                .short("E")
                .long("ex")
                .help("experiment name")
                .default_value("default")
                .takes_value(true),
        );
    // let mut help = Vec::new();
    // app.write_help(&mut help).expect("error on write app help");
    // let help = String::from_utf8(help).expect("failed to convert help Vec<u8> to String");
    let matches = app.get_matches();

    // parse arguments
    let train_config = TrainConfig::try_from(&matches).unwrap_or_default();

    run(train_config)
}

fn run(train_config: TrainConfig) -> Result<()> {
    println!("TrainConfig: {:#?}", train_config);

    let TrainConfig {
        visible_device_list,
        log_device_placement,
        batch_size_trn,
        batch_size_dev,
        lr,
        epochs,
        steps,
        eval_interval,
        ckpt_dir,
        ckpt,
        ex,
    } = train_config;

    let logdir = format!("./logs/{}", ex);

    // resume from checkpoint
    log::debug!("ckpt: {:?}", ckpt);
    let ckpt = ckpt.map(|x| dbg!(Ckpt::from_str(&x)).ok());
    let ckpt = match ckpt {
        Some(Some(x)) => Some(x),
        _ => None,
    };

    let epoch_start = {
        if let Some(ref ckpt) = ckpt {
            ckpt.epoch
        } else {
            1
        }
    };
    let steps: i64 = steps.unwrap_or(TRAIN_SIZE / batch_size_trn);
    let repeat_num = epochs * steps;
    let batch_size_trn_tensor = Tensor::<i64>::new(&[]).with_values(&[batch_size_trn])?;
    let batch_size_dev_tensor = Tensor::<i64>::new(&[]).with_values(&[batch_size_dev])?;
    let repeat_num_tensor = Tensor::<i64>::new(&[]).with_values(&[repeat_num])?;
    let lr_tensor = Tensor::<f32>::new(&[]).with_values(&[lr])?;

    log::debug!(
        "batch_size_trn: {}\tbatch_size_dev: {}\tlr: {}\tsteps: {}\trepeat_num: {}",
        batch_size_trn_tensor,
        batch_size_dev_tensor,
        lr_tensor,
        steps,
        repeat_num_tensor
    );

    // graph dir
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
    // let config_proto = include_bytes!("../assets/config.proto");
    // eprintln!("config_proto: {:?}", config_proto);

    let config_proto = build_config_proto(visible_device_list, log_device_placement)?;
    let mut sess_opt = SessionOptions::new();
    sess_opt.set_config(&config_proto.as_slice())?;
    let session = Session::new(&sess_opt, &graph)?;

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
    let _op_batch_data_val = graph.operation_by_name_required("data_val/dataset_val/get_next")?;
    let op_val_record_path = graph.operation_by_name_required("data_val/dataset_val/Const")?;
    // based on current dir
    let val_record_path: Tensor<String> = Tensor::from(String::from("assets/symptom_val.tfrecord"));

    let op_batch_data_test_init =
        graph.operation_by_name_required("data_test/dataset_test/MakeIterator")?;
    let _op_batch_data_test =
        graph.operation_by_name_required("data_test/dataset_test/get_next")?;
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
    let _file_path_tensor: Tensor<String> = Tensor::from(String::from("checkpoints/saved.ckpt"));

    // summary, not used here
    let _op_summary = graph.operation_by_name_required("summaries/summary_op/summary_op")?;

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
    let _data_closure = || -> Result<()> {
        let mut inspect_data_step = SessionRunArgs::new();
        // inspect_data_step.add_target(&op_batch_data_trn_init);
        let batch_data_ix = inspect_data_step.request_fetch(&op_batch_data_trn, 0);

        for _ in 0..3 {
            session.run(&mut inspect_data_step)?;
            let batch_data: Tensor<i64> = inspect_data_step.fetch(batch_data_ix)?;
            // eprintln!("batch_data:\n\t{:?}", batch_data);
            log::debug!("batch_data:\n\t{}", batch_data);
        }
        Ok(())
    };
    // _data_closure()?;

    // section Save
    let save = |ckpt: Ckpt| -> Result<()> {
        let checkpoint_path = format!("{}/{}", ckpt_dir, ckpt);
        let file_path_tensor: Tensor<String> = Tensor::from(checkpoint_path);
        let mut save_step = SessionRunArgs::new();
        save_step.add_feed(&op_file_path, 0, &file_path_tensor);
        save_step.add_target(&op_save);
        session.run(&mut save_step)?;
        Ok(())
    };

    // section Restore
    let restore = |init_step: &mut SessionRunArgs, ckpt: Ckpt| -> Result<()> {
        let checkpoint_path = format!("{}/{}", ckpt_dir, ckpt);
        session.run(init_step)?;
        // Load the model.
        let op_load = graph.operation_by_name_required("save/restore_all")?;
        let file_path_tensor: Tensor<String> = Tensor::from(checkpoint_path);
        let mut step = SessionRunArgs::new();
        step.add_feed(&op_file_path, 0, &file_path_tensor);
        step.add_target(&op_load);
        session.run(&mut step)?;
        Ok(())
    };

    // Train the model.
    let mut train = || -> Result<()> {
        // restore?
        log::debug!("ckpt {:?}", ckpt);
        if let Some(ckpt) = ckpt {
            log::info!("restore from ckpt: {}", ckpt);
            restore(&mut init_step, ckpt)?;
        }

        let mut writer = SummaryWriter::new(&logdir);
        let mut train_step = SessionRunArgs::new();
        let loss_token = train_step.request_fetch(&op_loss, 0);
        let global_step_token = train_step.request_fetch(&op_global_step, 0);
        train_step.add_target(&op_optimize);

        let eval = |dev: Dev, writer: &mut SummaryWriter, global_step: i64| -> Result<EvalValue> {
            log::info!("eval....");

            let mut eval_init_step = SessionRunArgs::new();
            eval_init_step.add_feed(&op_ph_batch_size_dev, 0, &batch_size_dev_tensor);

            let mut eval_step = SessionRunArgs::new();
            eval_step.add_feed(&op_ph_batch_size_dev, 0, &batch_size_dev_tensor);

            let (steps, rank_token, hit1_token, hit3_token, hit10_token) = match dev {
                Dev::Val => {
                    eval_step.add_feed(&op_val_record_path, 0, &val_record_path);
                    // bug
                    eval_init_step.add_target(&op_batch_data_val_init);
                    eval_step.add_target(&op_eval_val);
                    (
                        (VAL_SIZE / batch_size_dev) as usize,
                        eval_step.request_fetch(&op_rank_val, 0),
                        eval_step.request_fetch(&op_hit1_val, 0),
                        eval_step.request_fetch(&op_hit3_val, 0),
                        eval_step.request_fetch(&op_hit10_val, 0),
                    )
                }
                Dev::Test => {
                    eval_step.add_feed(&op_test_record_path, 0, &test_record_path);
                    eval_init_step.add_target(&op_batch_data_test_init);
                    eval_step.add_target(&op_eval_test);
                    (
                        (TEST_SIZE / batch_size_dev) as usize,
                        eval_step.request_fetch(&op_rank_test, 0),
                        eval_step.request_fetch(&op_hit1_test, 0),
                        eval_step.request_fetch(&op_hit3_test, 0),
                        eval_step.request_fetch(&op_hit10_test, 0),
                    )
                }
            };
            let mut ranks = Vec::with_capacity(steps);
            let mut hit1s = Vec::with_capacity(steps);
            let mut hit3s = Vec::with_capacity(steps);
            let mut hit10s = Vec::with_capacity(steps);

            session.run(&mut eval_init_step)?;
            for i in 0..steps {
                log::info!("eval [{}]", i);
                session.run(&mut eval_step)?;
                // fetch
                let rank: f32 = eval_step.fetch(rank_token)?[0];
                let hit1: f32 = eval_step.fetch(hit1_token)?[0];
                let hit3: f32 = eval_step.fetch(hit3_token)?[0];
                let hit10: f32 = eval_step.fetch(hit10_token)?[0];
                ranks.push(rank);
                hit1s.push(hit1);
                hit3s.push(hit3);
                hit10s.push(hit10);
            }
            let rank: f32 = ranks.into_iter().sum::<f32>() / steps as f32;
            let hit1: f32 = hit1s.into_iter().sum::<f32>() / steps as f32;
            let hit3: f32 = hit3s.into_iter().sum::<f32>() / steps as f32;
            let hit10: f32 = hit10s.into_iter().sum::<f32>() / steps as f32;
            writer.add_scalar("rank", rank, global_step as usize);
            writer.add_scalar("hit1", hit1, global_step as usize);
            writer.add_scalar("hit3", hit3, global_step as usize);
            writer.add_scalar("hit10", hit10, global_step as usize);
            log::info!(
                "rank: {}, hit1: {} hit3: {}, hit10: {}",
                rank,
                hit1,
                hit3,
                hit10
            );
            writer.flush();
            let value = EvalValue::new(rank, hit1, hit3, hit10);
            Ok(value)
        };

        let mut global_step: i64 = 0;
        // for e in 1..=epochs {
        for e in epoch_start..=epochs {
            // for i in 0..steps {
            let mut total_loss: f32 = 0.;
            for _ in 0..steps {
                session.run(&mut train_step)?;
                let loss: f32 = train_step.fetch(loss_token)?[0];
                total_loss += loss;
                // eprintln!("loss:\n\t{}", loss);
                global_step = train_step.fetch(global_step_token)?[0];
                writer.add_scalar("loss", loss, global_step as usize);
            }
            let loss = total_loss / steps as f32;
            log::info!("[{}] loss: {}", e, loss);
            if e % eval_interval == 0 {
                let eval_value = eval(Dev::Val, &mut writer, global_step)?;
                let ckpt = Ckpt::new(eval_value.rank, loss, e, global_step);
                // section save
                save(ckpt)?;
            }
        }
        writer.flush();
        Ok(())
    };
    train()?;

    Ok(())
}
