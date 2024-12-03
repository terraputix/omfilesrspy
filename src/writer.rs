use numpy::PyReadonlyArray2;
use omfiles_rs::{core::compression::CompressionType, io::writer::OmFileWriter};
use pyo3::prelude::*;
use std::rc::Rc;

pub fn write_om_file<'py>(
    file_path: &str,
    data: PyReadonlyArray2<'py, f32>,
    dim0: usize,
    dim1: usize,
    chunk0: usize,
    chunk1: usize,
    scalefactor: f32,
) -> PyResult<()> {
    // FIXME: We don't want to copy data around!!!
    let array = data.as_array();
    // Convert Python sequence to Vec<f32>
    let data_vec: Rc<Vec<f32>> = Rc::new(array.iter().copied().collect());

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

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::Array2, PyArray2, PyArrayMethods};
    use std::fs;

    #[test]
    fn test_write_om_file() {
        // Initialize Python
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Create test data
            let data = Array2::from_shape_fn((10, 20), |(i, j)| (i + j) as f32);
            let py_array = PyArray2::from_array(py, &data);

            // Test parameters
            let file_path = "test_data.om";
            let dim0 = 10;
            let dim1 = 20;
            let chunk0 = 5;
            let chunk1 = 5;
            let scalefactor = 1.0;

            // Write data
            let result = write_om_file(
                file_path,
                py_array.readonly(),
                dim0,
                dim1,
                chunk0,
                chunk1,
                scalefactor,
            );

            assert!(result.is_ok());
            assert!(fs::metadata(file_path).is_ok());

            // Clean up
            fs::remove_file(file_path).unwrap();
        });
    }
}
