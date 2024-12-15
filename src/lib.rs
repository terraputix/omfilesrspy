use numpy::{ndarray::Dim, PyArray, PyReadonlyArray2};
use pyo3::prelude::*;
mod errors;
mod reader;
mod test_utils;
mod writer;

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn omfilesrspy<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    // read from an om file
    #[pyfn(m)]
    #[pyo3(name = "read_om_file")]
    fn read_om_file<'py>(
        py: Python<'py>,
        file_path: &str,
        dim0_start: u64,
        dim0_end: u64,
        dim1_start: u64,
        dim1_end: u64,
    ) -> PyResult<Bound<'py, PyArray<f32, Dim<[usize; 2]>>>> {
        reader::read_om_file(py, file_path, dim0_start, dim0_end, dim1_start, dim1_end)
    }

    // write to an om file
    #[pyfn(m)]
    #[pyo3(name = "write_om_file")]
    fn write_om_file<'py>(
        file_path: &str,
        data: PyReadonlyArray2<'py, f32>,
        dim0: u64,
        dim1: u64,
        chunk0: u64,
        chunk1: u64,
        scale_factor: f32,
        add_offset: f32,
    ) -> PyResult<()> {
        writer::write_om_file(
            file_path,
            data,
            dim0,
            dim1,
            chunk0,
            chunk1,
            scale_factor,
            add_offset,
        )
    }

    Ok(())
}
