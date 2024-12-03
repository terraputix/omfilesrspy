use numpy::{ndarray::Dim, IntoPyArray, PyArray};
use omfiles_rs::io::reader::OmFileReader;
use pyo3::prelude::*;

pub fn read_om_file<'py>(
    py: Python<'py>,
    file_path: &str,
    dim0_start: usize,
    dim0_end: usize,
    dim1_start: usize,
    dim1_end: usize,
) -> Bound<'py, PyArray<f32, Dim<[usize; 1]>>> {
    let reader = OmFileReader::from_file(file_path).unwrap();
    let data = reader
        .read_range(Some(dim0_start..dim0_end), Some(dim1_start..dim1_end))
        .unwrap();

    data.into_pyarray(py)
}
