use crate::{
    array_index::ArrayIndex, data_type::to_numpy_dtype, errors::convert_omfilesrs_error,
    fsspec_backend::FsSpecBackend,
};
use delegate::delegate;
use num_traits::Zero;
use numpy::{Element, IntoPyArray, PyArrayMethods, PyUntypedArray};
use omfiles_rs::{
    backend::{backends::OmFileReaderBackend, mmapfile::MmapFile},
    core::data_types::{DataType, OmFileArrayDataType, OmFileScalarDataType},
    io::{reader::OmFileReader, writer::OmOffsetSize},
};
use pyo3::{prelude::*, BoundObject};
use std::{collections::HashMap, sync::Arc};

#[pyclass]
pub struct OmFilePyReader {
    reader: OmFileReader<BackendImpl>,
    #[pyo3(get)]
    shape: Vec<u64>,
}

unsafe impl Send for OmFilePyReader {}
unsafe impl Sync for OmFilePyReader {}

#[pymethods]
impl OmFilePyReader {
    #[new]
    fn new(source: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            if let Ok(path) = source.extract::<String>(py) {
                // If source is a string, treat it as a file path
                Self::from_path(&path)
            } else {
                let obj = source.bind(py);
                if obj.hasattr("read")? && obj.hasattr("seek")? && obj.hasattr("fs")? {
                    // If source has fsspec-like attributes, treat it as a fsspec file object
                    Self::from_fsspec(source)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Input must be either a file path string or a fsspec file object",
                    ))
                }
            }
        })
    }

    #[staticmethod]
    fn from_path(file_path: &str) -> PyResult<Self> {
        use omfiles_rs::backend::mmapfile::Mode;
        use std::fs::File;

        let file_handle = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let backend = BackendImpl::Mmap(MmapFile::new(file_handle, Mode::ReadOnly)?);
        let reader = OmFileReader::new(Arc::new(backend)).map_err(convert_omfilesrs_error)?;
        let shape = reader.get_dimensions().to_vec();

        Ok(Self { reader, shape })
    }

    #[staticmethod]
    fn from_fsspec(file_obj: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let bound_object = file_obj.bind(py);

            if !bound_object.hasattr("read")?
                || !bound_object.hasattr("seek")?
                || !bound_object.hasattr("fs")?
            {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Input must be a valid fsspec file object with read, seek methods and fs attribute",
                    ));
            }

            let backend = BackendImpl::FsSpec(FsSpecBackend::new(file_obj)?);
            let reader = OmFileReader::new(Arc::new(backend)).map_err(convert_omfilesrs_error)?;
            let shape = reader.get_dimensions().to_vec();

            Ok(Self { reader, shape })
        })
    }

    fn get_flat_variable_metadata(&self) -> PyResult<HashMap<String, (u64, u64)>> {
        let metadata = self.reader.get_flat_variable_metadata();
        Ok(metadata
            .into_iter()
            .map(|(key, offset_size)| (key, (offset_size.offset, offset_size.size)))
            .collect())
    }

    fn init_from_offset_size(&self, offset: u64, size: u64) -> PyResult<Self> {
        let reader = self
            .reader
            .init_child_from_offset_size(OmOffsetSize::new(offset, size))
            .map_err(convert_omfilesrs_error)?;

        let shape = reader.get_dimensions().to_vec();
        Ok(Self { reader, shape })
    }

    #[getter]
    fn dtype(&self) -> PyResult<String> {
        Ok(to_numpy_dtype(&self.reader.data_type()).to_string())
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self.reader.get_name().unwrap_or("".to_string()))
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        ranges: ArrayIndex,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let read_ranges = ranges.to_read_range(&self.shape)?;

        let reader = &self.reader;
        let dtype = reader.data_type();

        let scalar_error =
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scalar data types are not supported");

        let untyped_py_array_or_error = match dtype {
            DataType::None => Err(scalar_error),
            DataType::Int8 => Err(scalar_error),
            DataType::Uint8 => Err(scalar_error),
            DataType::Int16 => Err(scalar_error),
            DataType::Uint16 => Err(scalar_error),
            DataType::Int32 => Err(scalar_error),
            DataType::Uint32 => Err(scalar_error),
            DataType::Int64 => Err(scalar_error),
            DataType::Uint64 => Err(scalar_error),
            DataType::Float => Err(scalar_error),
            DataType::Double => Err(scalar_error),
            DataType::String => Err(scalar_error),
            DataType::Int8Array => read_untyped_array::<i8>(&reader, read_ranges, py),
            DataType::Uint8Array => read_untyped_array::<u8>(&reader, read_ranges, py),
            DataType::Int16Array => read_untyped_array::<i16>(&reader, read_ranges, py),
            DataType::Uint16Array => read_untyped_array::<u16>(&reader, read_ranges, py),
            DataType::Int32Array => read_untyped_array::<i32>(&reader, read_ranges, py),
            DataType::Uint32Array => read_untyped_array::<u32>(&reader, read_ranges, py),
            DataType::Int64Array => read_untyped_array::<i64>(&reader, read_ranges, py),
            DataType::Uint64Array => read_untyped_array::<u64>(&reader, read_ranges, py),
            DataType::FloatArray => read_untyped_array::<f32>(&reader, read_ranges, py),
            DataType::DoubleArray => read_untyped_array::<f64>(&reader, read_ranges, py),
            DataType::StringArray => {
                unimplemented!("String arrays are currently not implemented")
            }
        };

        let untyped_py_array = untyped_py_array_or_error?;

        return Ok(untyped_py_array);
    }

    fn get_scalar(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| match self.reader.data_type() {
            DataType::Int8 => self.read_scalar_value::<i8>(py),
            DataType::Uint8 => self.read_scalar_value::<u8>(py),
            DataType::Int16 => self.read_scalar_value::<i16>(py),
            DataType::Uint16 => self.read_scalar_value::<u16>(py),
            DataType::Int32 => self.read_scalar_value::<i32>(py),
            DataType::Uint32 => self.read_scalar_value::<u32>(py),
            DataType::Int64 => self.read_scalar_value::<i64>(py),
            DataType::Uint64 => self.read_scalar_value::<u64>(py),
            DataType::Float => self.read_scalar_value::<f32>(py),
            DataType::Double => self.read_scalar_value::<f64>(py),
            DataType::String => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "String data type is not supported",
            )),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data type is not scalar",
            )),
        })
    }
}

impl OmFilePyReader {
    fn read_scalar_value<'py, T>(&self, py: Python<'py>) -> PyResult<PyObject>
    where
        T: Element + OmFileScalarDataType + IntoPyObject<'py>,
    {
        let value = self.reader.read_scalar::<T>();

        value
            .into_pyobject(py)
            .map(BoundObject::into_any)
            .map(BoundObject::unbind)
            .map_err(Into::into)
    }
}

fn read_untyped_array<'py, T: Element + OmFileArrayDataType + Clone + Zero>(
    reader: &OmFileReader<impl OmFileReaderBackend>,
    read_ranges: Vec<std::ops::Range<u64>>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let array = reader
        .read::<T>(&read_ranges, None, None)
        .map_err(convert_omfilesrs_error)?;
    // We only add dimensions that are no singleton dimensions to the output shape
    // This is basically a dimensional squeeze and it is the same behavior as numpy
    Ok(array.squeeze().into_pyarray(py).as_untyped().to_owned()) // FIXME: avoid cloning?
}

/// Concrete wrapper type for the backend implementation, delegating to the appropriate backend
enum BackendImpl {
    Mmap(MmapFile),
    FsSpec(FsSpecBackend),
}

impl OmFileReaderBackend for BackendImpl {
    delegate! {
        to match self {
            BackendImpl::Mmap(backend) => backend,
            BackendImpl::FsSpec(backend) => backend,
        } {
            fn count(&self) -> usize;
            fn needs_prefetch(&self) -> bool;
            fn prefetch_data(&self, offset: usize, count: usize);
            fn pre_read(&self, offset: usize, count: usize) -> Result<(), omfiles_rs::errors::OmFilesRsError>;
            fn get_bytes(&self, offset: u64, count: u64) -> Result<&[u8], omfiles_rs::errors::OmFilesRsError>;
            fn get_bytes_owned(&self, offset: u64, count: u64) -> Result<Vec<u8>, omfiles_rs::errors::OmFilesRsError>;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_index::IndexType;
    use crate::create_test_binary_file;
    use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};

    #[test]
    fn test_read_simple_v3_data() -> Result<(), Box<dyn std::error::Error>> {
        create_test_binary_file!("read_test.om")?;
        let file_path = "test_files/read_test.om";
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let reader = OmFilePyReader::from_path(file_path).unwrap();
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

            assert_eq!(data.shape(), [5, 5]);

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
