use numpy::PyReadonlyArray2;
use omfiles_rs::{core::compression::CompressionType, io::writer2::OmFileWriter2};
use pyo3::prelude::*;
use std::fs::File;

use crate::errors::convert_omfilesrs_error;

pub fn write_om_file<'py>(
    file_path: &str,
    data: PyReadonlyArray2<'py, f32>,
    dim0: u64,
    dim1: u64,
    chunk0: u64,
    chunk1: u64,
    scale_factor: f32,
    add_offset: f32,
) -> PyResult<()> {
    let initial_capacity = data.len()?;
    let file_handle = File::create(file_path)?;
    let mut file_writer = OmFileWriter2::new(&file_handle, initial_capacity as u64);

    let mut writer = file_writer
        .prepare_array::<f32>(
            vec![dim0, dim1],
            vec![chunk0, chunk1],
            CompressionType::P4nzdec256,
            scale_factor,
            add_offset,
            256,
        )
        .map_err(convert_omfilesrs_error)?;

    let array = data.as_slice().expect("No array found behind `data`");

    writer
        .write_data(array, None, None, None)
        .map_err(convert_omfilesrs_error)?;

    let variable_meta = writer.finalize();
    let variable = file_writer
        .write_array(variable_meta, "data", &[])
        .map_err(convert_omfilesrs_error)?;
    file_writer
        .write_trailer(variable)
        .map_err(convert_omfilesrs_error)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::Array2, PyArray2, PyArrayMethods};
    use std::fs;

    #[test]
    fn test_write_om_file() -> Result<(), Box<dyn std::error::Error>> {
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
            let scale_factor = 1.0;
            let add_offset = 0.0;

            // Write data
            let result = write_om_file(
                file_path,
                py_array.readonly(),
                dim0,
                dim1,
                chunk0,
                chunk1,
                scale_factor,
                add_offset,
            );

            assert!(result.is_ok());
            assert!(fs::metadata(file_path).is_ok());

            // Clean up
            fs::remove_file(file_path).unwrap();
        });

        Ok(())
    }
}
