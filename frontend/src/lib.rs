use wasm_bindgen::prelude::*;

pub mod app;
mod bindings;
mod pages;
mod utils;

#[wasm_bindgen]
pub fn run() {
    app::run();
}
