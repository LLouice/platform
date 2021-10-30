use clap::Clap;

/// Simple program to greet a person
#[derive(Clap, Debug)]
#[clap(name = "hello")]
struct Hello {
    /// Name of the person to greet
    #[clap(short, long)]
    name: String,

    /// Number of times to greet
    #[clap(short, long, default_value = "1")]
    count: u8,
}

fn main() {
    let hello = Hello::parse();

    for _ in 0..hello.count {
        println!("Hello {}!", hello.name)
    }
}
