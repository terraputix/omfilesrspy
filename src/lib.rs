use numpy::{ndarray::Dim, PyArray, PyReadonlyArray2};
use pyo3::prelude::*;
mod reader;
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
        dim0_start: usize,
        dim0_end: usize,
        dim1_start: usize,
        dim1_end: usize,
    ) -> Bound<'py, PyArray<f32, Dim<[usize; 1]>>> {
        reader::read_om_file(py, file_path, dim0_start, dim0_end, dim1_start, dim1_end)
    }

    // write to an om file
    #[pyfn(m)]
    #[pyo3(name = "write_om_file")]
    fn write_om_file<'py>(
        file_path: &str,
        data: PyReadonlyArray2<'py, f32>,
        dim0: usize,
        dim1: usize,
        chunk0: usize,
        chunk1: usize,
        scalefactor: f32,
    ) -> PyResult<()> {
        writer::write_om_file(file_path, data, dim0, dim1, chunk0, chunk1, scalefactor)
    }

    Ok(())
}
