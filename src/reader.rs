use crate::{
    array_index::ArrayIndex, errors::convert_omfilesrs_error, fsspec_backend::FsSpecBackend,
};
use numpy::{ndarray::ArrayD, Element, IntoPyArray, PyArrayMethods, PyUntypedArray};
use omfiles_rs::{
    backend::{backends::OmFileReaderBackend, mmapfile::MmapFile},
    core::data_types::OmFileArrayDataType,
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
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let read_ranges = ranges.to_read_range(self.get_shape())?;
        // We only add dimensions that are no singleton dimensions to the output shape
        // This is basically a dimensional squeeze and it is the same behavior as numpy
        let output_shape = read_ranges
            .iter()
            .map(|range| (range.end - range.start) as usize)
            .filter(|&size| size != 1)
            .collect::<Vec<_>>();

        let reader = self.get_reader();
        let dtype = reader.data_type();

        let scalar_error =
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scalar data types are not supported");

        let untyped_py_array_or_error = match dtype {
            omfiles_rs::core::data_types::DataType::None => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Int8 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Uint8 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Int16 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Uint16 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Int32 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Uint32 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Int64 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Uint64 => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Float => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Double => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::String => Err(scalar_error),
            omfiles_rs::core::data_types::DataType::Int8Array => {
                read_untyped_array::<i8>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::Uint8Array => {
                read_untyped_array::<u8>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::Int16Array => {
                read_untyped_array::<i16>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::Uint16Array => {
                read_untyped_array::<u16>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::Int32Array => {
                read_untyped_array::<i32>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::Uint32Array => {
                read_untyped_array::<u32>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::Int64Array => {
                read_untyped_array::<i64>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::Uint64Array => {
                read_untyped_array::<u64>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::FloatArray => {
                read_untyped_array::<f32>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::DoubleArray => {
                read_untyped_array::<f64>(&reader, read_ranges, output_shape, py)
            }
            omfiles_rs::core::data_types::DataType::StringArray => {
                unimplemented!("String arrays are currently not implemented")
            }
        };

        let untyped_py_array = untyped_py_array_or_error?;

        return Ok(untyped_py_array);
    }
}

fn read_untyped_array<'py, T: Element + OmFileArrayDataType>(
    reader: &OmFileReader<impl OmFileReaderBackend>,
    read_ranges: Vec<std::ops::Range<u64>>,
    output_shape: Vec<usize>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let flat_data = reader
        .read::<T>(&read_ranges, None, None)
        .map_err(convert_omfilesrs_error)?;
    let array = ArrayD::from_shape_vec(output_shape, flat_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(array.into_pyarray(py).as_untyped().to_owned()) // FIXME: avoid cloning?
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
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
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
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
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
    use numpy::{PyArrayDyn, PyArrayMethods};

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
            let data = data
                .downcast::<PyArrayDyn<f32>>()
                .expect("Could not downcast to PyArrayDyn<f32>");
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
