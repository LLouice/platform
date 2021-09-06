pub mod app;
mod bindings;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn run() {
    app::run();
}
