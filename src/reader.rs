use crate::errors::convert_omfilesrs_error;
use numpy::{
    ndarray::{Array2, Dim},
    IntoPyArray, PyArray,
};
use omfiles_rs::io::reader2::OmFileReader2;
use pyo3::prelude::*;

pub fn read_om_file<'py>(
    py: Python<'py>,
    file_path: &str,
    dim0_start: u64,
    dim0_end: u64,
    dim1_start: u64,
    dim1_end: u64,
) -> PyResult<Bound<'py, PyArray<f32, Dim<[usize; 2]>>>> {
    let reader = OmFileReader2::from_file(file_path).map_err(convert_omfilesrs_error)?;
    let flat_data = reader
        .read_simple(&[dim0_start..dim0_end, dim1_start..dim1_end], None, None)
        .map_err(convert_omfilesrs_error)?;

    let rows = (dim0_end - dim0_start) as usize;
    let cols = (dim1_end - dim1_start) as usize;

    // Create a 2D array from the flat data
    let array = Array2::from_shape_vec((rows, cols), flat_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(array.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_binary_file;
    use numpy::PyArrayMethods;

    #[test]
    fn test_read_simple_v3_data() -> Result<(), Box<dyn std::error::Error>> {
        create_binary_file(
            "read_test.om",
            &[
                79, 77, 3, 0, 4, 130, 0, 2, 3, 34, 0, 4, 194, 2, 10, 4, 178, 0, 12, 4, 242, 0, 14,
                197, 17, 20, 194, 2, 22, 194, 2, 24, 3, 3, 228, 200, 109, 1, 0, 0, 20, 0, 4, 0, 0,
                0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 128, 63, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
                0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 100, 97, 116, 97, 0, 0, 0, 0, 79, 77, 3, 0,
                0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0,
            ],
        )?;

        let file_path = "test_files/read_test.om";
        let dim0_start = 0;
        let dim0_end = 5;
        let dim1_start = 0;
        let dim1_end = 5;

        // Initialize Python
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let data =
                read_om_file(py, file_path, dim0_start, dim0_end, dim1_start, dim1_end).unwrap();
            let read_only = data.readonly();
            let data = read_only.as_slice().unwrap();
            let expected_data = vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ];
            assert_eq!(data, expected_data);
        });

        Ok(())
    }
}
