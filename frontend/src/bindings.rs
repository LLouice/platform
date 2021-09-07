use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/static/e.js")]
extern "C" {
    #[wasm_bindgen(js_name = "getEchartInstance")]
    pub fn get_element_by_id(selector: String) -> JsValue;

    #[wasm_bindgen(js_name = "main")]
    pub fn main();

    #[wasm_bindgen(js_name = "displayNetwork")]
    pub async fn display_network(src_type: String, name: String);

}
