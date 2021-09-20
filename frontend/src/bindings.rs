use js_sys::Function;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/static/e.js")]
extern "C" {
    #[wasm_bindgen(js_name = "getEchartInstance")]
    pub fn get_element_by_id(selector: String) -> JsValue;

    #[wasm_bindgen(js_name = "displayNetwork")]
    pub async fn display_network(src_type: String, name: String);

    #[wasm_bindgen(js_name = "displayWordCloud")]
    pub fn display_word_cloud(data: Option<js_sys::Array>);

    #[wasm_bindgen(js_name = "getOption")]
    pub fn get_echarts_option(selector: Option<String>) -> JsValue;

    // #[wasm_bindgen(js_name = "setOption")]
    // pub fn set_echarts_option(chart: &JsValue, opt: &JsValue);

    #[wasm_bindgen(js_name = "main")]
    pub async fn main();

    #[wasm_bindgen(js_name = "main_with_cb")]
    pub fn main_with_cb(f: Function);
}
