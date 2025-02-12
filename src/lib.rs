use pyo3::prelude::*;
mod array_index;
mod compression;
mod data_type;
mod errors;
mod fsspec_backend;
mod hierarchy;
mod reader;
mod test_utils;
mod writer;

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn omfilesrspy<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<reader::OmFilePyReader>()?;
    m.add_class::<writer::OmFilePyWriter>()?;

    Ok(())
}
