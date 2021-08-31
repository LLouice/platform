use platform::kg::Kg;
use tokio;

#[tokio::main]
async fn test_kg2() {
    let src_type = "Symptom";
    let name = "肩背痛";
    let res = Kg::convert(src_type, name, Kg::get_out_links(src_type, name).await).unwrap();
    println!("{:?}", res);
}

#[test]
fn test_kg() {
    test_kg2();
}
