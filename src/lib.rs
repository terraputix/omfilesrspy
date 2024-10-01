use omfiles_rs::om::reader::OmFileReader;
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

    Ok(())
}
