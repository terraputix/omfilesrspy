use numpy::{ndarray::Dim, IntoPyArray, PyArray, PyReadonlyArray2};
use pyo3::prelude::*;

use omfiles_rs::{
    core::compression::CompressionType,
    io::{reader::OmFileReader, writer::OmFileWriter},
};
use std::rc::Rc;

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
        let reader = OmFileReader::from_file(file_path).unwrap();
        let data = reader
            .read_range(Some(dim0_start..dim0_end), Some(dim1_start..dim1_end))
            .unwrap();

        data.into_pyarray(py)
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
        // Convert Python sequence to Vec<f32>
        let data_vec: Rc<Vec<f32>> = Rc::new(data.extract()?);

        let writer = OmFileWriter::new(dim0, dim1, chunk0, chunk1);

        writer
            .write_all_to_file(
                file_path,
                CompressionType::P4nzdec256,
                scalefactor,
                data_vec,
                true,
            )
            .unwrap();

        Ok(())
    }

    Ok(())
}
