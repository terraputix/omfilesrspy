use crate::{
    array_index::ArrayIndex, errors::convert_omfilesrs_error, fsspec_backend::FsSpecBackend,
};
use numpy::{ndarray::ArrayD, IntoPyArray, PyArrayDyn};
use omfiles_rs::{
    backend::{backends::OmFileReaderBackend, mmapfile::MmapFile},
    io::reader::OmFileReader,
};
use pyo3::prelude::*;
use std::sync::Arc;

// Reader trait for common functionality
trait OmFilePyReaderTrait {
    fn get_reader(&self) -> &OmFileReader<impl OmFileReaderBackend>;
    fn get_shape(&self) -> &Vec<u64>;

    fn get_item<'py>(
        &self,
        py: Python<'py>,
        ranges: ArrayIndex,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let read_ranges = ranges.to_read_range(self.get_shape())?;
        // We only add dimensions that are no singleton dimensions to the output shape
        // This is basically a dimensional squeeze and it is the same behavior as numpy
        let output_shape = read_ranges
            .iter()
            .map(|range| (range.end - range.start) as usize)
            .filter(|&size| size != 1)
            .collect::<Vec<_>>();

        let flat_data = self
            .get_reader()
            .read::<f32>(&read_ranges, None, None)
            .map_err(convert_omfilesrs_error)?;

        let array = ArrayD::from_shape_vec(output_shape, flat_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(array.into_pyarray(py))
    }
}

#[pyclass]
pub struct OmFilePyReader {
    reader: OmFileReader<MmapFile>,
    shape: Vec<u64>,
}
unsafe impl Send for OmFilePyReader {}
unsafe impl Sync for OmFilePyReader {}

#[pymethods]
impl OmFilePyReader {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        let reader = OmFileReader::from_file(file_path).map_err(convert_omfilesrs_error)?;
        let shape = reader.get_dimensions().to_vec();

        Ok(Self { reader, shape })
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        ranges: ArrayIndex,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.get_item(py, ranges)
    }
}

impl OmFilePyReaderTrait for OmFilePyReader {
    fn get_reader(&self) -> &OmFileReader<impl OmFileReaderBackend> {
        &self.reader
    }

    fn get_shape(&self) -> &Vec<u64> {
        &self.shape
    }
}

#[pyclass]
pub struct OmFilePyFsSpecReader {
    reader: OmFileReader<FsSpecBackend>,
    shape: Vec<u64>,
}
unsafe impl Send for OmFilePyFsSpecReader {}
unsafe impl Sync for OmFilePyFsSpecReader {}

#[pymethods]
impl OmFilePyFsSpecReader {
    #[new]
    fn new(file_obj: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| -> PyResult<Self> {
            let bound_object = file_obj.bind(py);

            if !bound_object.hasattr("read")?
                || !bound_object.hasattr("seek")?
                || !bound_object.hasattr("fs")?
            {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Input must be a valid fsspec file object with read, seek methods and fs attribute",
                ));
            }

            let backend = FsSpecBackend::new(file_obj)?;
            let reader = OmFileReader::new(Arc::new(backend)).map_err(convert_omfilesrs_error)?;
            let shape = reader.get_dimensions().to_vec();
            Ok(Self { reader, shape })
        })
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        ranges: ArrayIndex,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.get_item(py, ranges)
    }
}

impl OmFilePyReaderTrait for OmFilePyFsSpecReader {
    fn get_reader(&self) -> &OmFileReader<impl OmFileReaderBackend> {
        &self.reader
    }

    fn get_shape(&self) -> &Vec<u64> {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_index::IndexType;
    use crate::create_test_binary_file;
    use numpy::PyArrayMethods;

    #[test]
    fn test_read_simple_v3_data() -> Result<(), Box<dyn std::error::Error>> {
        create_test_binary_file!("read_test.om")?;
        let file_path = "test_files/read_test.om";
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
