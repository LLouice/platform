cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        pub mod data;
        pub use data::GraphData;
    } else {
        #[macro_use]
        extern crate anyhow;

        pub mod common;
        pub mod kg;
        pub mod neo4j;
        pub mod triple;
        pub mod utils;

        pub mod data;
        pub use data::GraphData;
    }
}
