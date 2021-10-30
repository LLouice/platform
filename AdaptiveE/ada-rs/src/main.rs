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
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn main() -> Result<()> {
    run()
}

fn run() -> Result<()> {
    let filename = "examples/regression_checkpoint/model.pb"; // y = w * x + b
    if !Path::new(filename).exists() {
        bail!(format!(
            "Run 'python regression_checkpoint.py' to generate \
                     {} and try again.",
            filename
        ));
    }

    // Generate some test data.
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

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;

    // session option
    let config_proto = include_bytes!("config.proto");
    eprintln!("config_proto: {:?}", config_proto);
    let mut sess_opt = SessionOptions::new();
    sess_opt.set_config(config_proto.as_slice())?;
    let session = Session::new(&sess_opt, &graph)?;

    // get op from graph_def
    let op_x = graph.operation_by_name_required("x")?;
    let op_y = graph.operation_by_name_required("y")?;
    let op_init = graph.operation_by_name_required("init")?;
    let op_train = graph.operation_by_name_required("train")?;
    let op_w = graph.operation_by_name_required("w")?;
    let op_b = graph.operation_by_name_required("b")?;
    let op_file_path = graph.operation_by_name_required("save/Const")?;
    let op_save = graph.operation_by_name_required("save/control_dependency")?;
    let file_path_tensor: Tensor<String> =
        Tensor::from(String::from("examples/regression_checkpoint/saved.ckpt"));

    // Load the test data into the session.
    let mut init_step = SessionRunArgs::new();
    init_step.add_target(&op_init);
    session.run(&mut init_step)?;

    // Train the model.
    let mut train_step = SessionRunArgs::new();
    train_step.add_feed(&op_x, 0, &x);
    train_step.add_feed(&op_y, 0, &y);
    train_step.add_target(&op_train);
    for _ in 0..steps {
        session.run(&mut train_step)?;
    }

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
    Ok(())
}

/*
mod py {
    use pyo3::types::PyList;
    use pyo3::{
        prelude::*,
        types::{IntoPyDict, PyModule},
    };

    fn register_path<P: AsRef<std::path::Path>>(py: Python, path: P) -> PyResult<()> {
        let syspath: &PyList = py.import("sys")?.getattr("path")?.downcast::<PyList>()?;
        let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let py_file = workspace_dir.join(path.as_ref());
        syspath.insert(0, py_file.to_str().unwrap())?;
        eprintln!("syspath: {:#?}", syspath);
        Ok(())
    }

    pub fn config() -> PyResult<Vec<u8>> {
        Python::with_gil(|py| -> PyResult<Vec<u8>> {
            register_path(py, "examples")?;

            let module = PyModule::import(py, "config").expect("failed to import `config` module");
            // let config_str = module.getattr("build_config_proto")?.extract::<Vec<u8>>()?;
            let config_proto = module
                .getattr("build_config_proto")?
                .call1((3,))?
                .extract::<Vec<u8>>()?;
            Ok(config_proto)
        })
    }

    pub fn import() -> PyResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            register_path(py, "examples")?;
            let module = PyModule::import(py, "import").expect("failed to import `import` module");
            module.getattr("foo")?.call0()?;

            Ok(())
        });
        Ok(())
    }
}
*/
