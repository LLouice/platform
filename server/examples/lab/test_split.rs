fn main() {
    let string = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
    let res = parse(string);
    println!("{:?}", res);
    let string = "http://www.w3.org/2000/01/症状相关药品";
    let res = parse(string);
    println!("{:?}", res);
}

fn parse<P: AsRef<str>>(string: P) -> Option<String> {
    let string = string.as_ref();
    if !string.contains("#") {
        string.split("/").last().map(|x| x.to_string())
    } else {
        None
    }
}
