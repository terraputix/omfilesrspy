//! omfiles-rs: A Rust library for working with Om files
//!
//! This library provides functionality for reading and writing Om file format.
//!
pub mod io {
    pub mod buffered_writer;
    pub mod reader;
    pub mod writer;
}

pub mod core {
    pub mod c_defaults;
    pub mod compression;
    pub mod data_types;
}

pub mod backend {
    pub mod backends;
    pub mod mmapfile;
}

pub mod errors;

mod utils;

#[cfg(feature = "python")]
mod python;
