pub mod app;
mod bindings;
mod pages;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn run() {
    app::run();
}
