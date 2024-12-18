use crate::{array_index::ArrayIndex, errors::convert_omfilesrs_error};
use numpy::{ndarray::ArrayD, IntoPyArray, PyArrayDyn};
use omfiles_rs::{backend::mmapfile::MmapFile, io::reader2::OmFileReader2};
use pyo3::prelude::*;

#[pyclass]
pub struct OmFilePyReader {
    reader: OmFileReader2<MmapFile>,
    shape: Vec<u64>,
}

unsafe impl Send for OmFilePyReader {}
unsafe impl Sync for OmFilePyReader {}

#[pymethods]
impl OmFilePyReader {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        let reader = OmFileReader2::from_file(file_path).map_err(convert_omfilesrs_error)?;
        let shape = reader.get_dimensions().to_vec();

        Ok(Self { reader, shape })
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        ranges: ArrayIndex,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let read_ranges = ranges.to_read_range(&self.shape)?;
        // We only add dimensions that are no singleton dimensions to the output shape
        // This is basically a dimensional squeeze and it is the same behavior as numpy
        let output_shape = read_ranges
            .iter()
            .map(|range| (range.end - range.start) as usize)
            .filter(|&size| size != 1)
            .collect::<Vec<_>>();

        let flat_data = self
            .reader
            .read_simple(&read_ranges, None, None)
            .map_err(convert_omfilesrs_error)?;

        let array = ArrayD::from_shape_vec(output_shape, flat_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(array.into_pyarray(py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array_index::IndexType, test_utils::create_binary_file};
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

        // Initialize Python
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let reader = OmFilePyReader::new(file_path).unwrap();
            let ranges = ArrayIndex(vec![
                IndexType::Slice {
                    start: Some(0),
                    stop: Some(5),
                    step: None,
                },
                IndexType::Slice {
                    start: Some(0),
                    stop: Some(5),
                    step: None,
                },
            ]);
            let data = reader.__getitem__(py, ranges).expect("Could not get item!");
            let read_only = data.readonly();
            let data = read_only.as_slice().expect("Could not convert to slice!");
            let expected_data = vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ];
            assert_eq!(data, expected_data);
        });

        Ok(())
    }
}
