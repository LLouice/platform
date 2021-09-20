#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]

cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        pub mod data;
        pub use data::GraphData;
    } else {
        pub mod common;
        pub mod kg;
        pub mod neo4j;
        pub mod triple;
        #[macro_use] pub mod utils;
        pub mod session;

        pub mod data;
        pub use data::GraphData;
    }
}
