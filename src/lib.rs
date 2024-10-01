use numpy::PyReadonlyArray2;

use omfiles_rs::{
    compression::CompressionType,
    om::{reader::OmFileReader, writer::OmFileWriter},
};
use pyo3::prelude::*;

#[pyfunction]
fn read_om_file(
    file_path: &str,
    dim0_start: usize,
    dim0_end: usize,
    dim1_start: usize,
    dim1_end: usize,
) -> PyResult<Vec<f32>> {
    let reader = OmFileReader::from_file(file_path).unwrap();
    let data = reader
        .read_range(Some(dim0_start..dim0_end), Some(dim1_start..dim1_end))
        .unwrap();
    Ok(data)
}

#[pyfunction]
fn write_om_file<'py>(
    file_path: &str,
    data: PyReadonlyArray2<'py, f32>,
    dim0: usize,
    dim1: usize,
    chunk0: usize,
    chunk1: usize,
) -> PyResult<()> {
    let data = data.as_slice().unwrap();
    let writer = OmFileWriter::new(dim0, dim1, chunk0, chunk1);

    writer
        .write_all_to_file(file_path, CompressionType::P4nzdec256, 10.0, &data, true)
        .unwrap();

    Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn omfilesrspy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    m.add_function(wrap_pyfunction!(read_om_file, m)?)?;

    m.add_function(wrap_pyfunction!(write_om_file, m)?)?;

    Ok(())
}
