use anyhow::Result;
use pyo3::types::PyList;
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyModule},
};

#[inline(always)]
fn set_py() {
    std::env::set_var("PYO3_PYTHON", "/home/llouice/anaconda3/envs/tf1/bin/python");
}

fn demo() -> PyResult<()> {
    Python::with_gil(|py| -> PyResult<()> {
        let activators = PyModule::from_code(
            py,
            r#"
def relu(x):
    """see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)"""
    return max(0.0, x)

def leaky_relu(x, slope=0.01):
    return x if x >= 0 else x * slope
    "#,
            "activators.py",
            "activators",
        )?;

        let relu_result: f64 = activators.getattr("relu")?.call1((-1.0,))?.extract()?;
        assert_eq!(relu_result, 0.0);

        let kwargs = [("slope", 0.2)].into_py_dict(py);
        let lrelu_result: f64 = activators
            .getattr("leaky_relu")?
            .call((-1.0,), Some(kwargs))?
            .extract()?;
        println!("lrelu_result: {}", lrelu_result);
        assert_eq!(lrelu_result, -0.2);
        Ok(())
    })
}

fn register_path<P: AsRef<std::path::Path>>(py: Python, path: P) -> PyResult<()> {
    let syspath: &PyList = py.import("sys")?.getattr("path")?.downcast::<PyList>()?;
    let workspace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let py_file = workspace_dir.join(path.as_ref());
    syspath.insert(0, py_file.to_str().unwrap())?;
    Ok(())
}

fn import() -> PyResult<()> {
    Python::with_gil(|py| -> PyResult<()> {
        register_path(py, "examples")?;
        let module = PyModule::import(py, "import").expect("failed to import `import` module");
        module.getattr("foo")?.call0()?;

        Ok(())
    });
    Ok(())
}

fn main() -> PyResult<()> {
    import()?;
    Ok(())
}
