use crate::{
    compression::PyCompressionType, errors::convert_omfilesrs_error, hierarchy::OmVariable,
};
use numpy::{
    dtype, Element, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArray, PyUntypedArrayMethods,
};
use omfiles_rs::{
    core::{
        compression::CompressionType,
        data_types::{OmFileArrayDataType, OmFileScalarDataType},
    },
    errors::OmFilesRsError,
    io::writer::{OmFileWriter, OmFileWriterArrayFinalized, OmOffsetSize},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fs::File;

#[pyclass]
pub struct OmFilePyWriter {
    file_writer: OmFileWriter<File>,
}

#[pymethods]
impl OmFilePyWriter {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        let file_handle = File::create(file_path)?;
        let writer = OmFileWriter::new(file_handle, 8 * 1024); // initial capacity of 8KB
        Ok(Self {
            file_writer: writer,
        })
    }

    #[pyo3(
            text_signature = "(root_variable, /)",
            signature = (root_variable)
        )]
    fn close(&mut self, root_variable: OmVariable) -> PyResult<()> {
        self.file_writer
            .write_trailer(root_variable.into())
            .map_err(convert_omfilesrs_error)
    }

    #[pyo3(
            text_signature = "(data, chunks, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d', name='data', children=[])",
            signature = (data, chunks, scale_factor=None, add_offset=None, compression=None, name=None, children=None)
        )]
    fn write_array(
        &mut self,
        data: &Bound<'_, PyUntypedArray>,
        chunks: Vec<u64>,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
        name: Option<&str>,
        children: Option<Vec<OmVariable>>,
    ) -> PyResult<OmVariable> {
        let name = name.unwrap_or("data");
        let children: Vec<OmOffsetSize> = children
            .unwrap_or_default()
            .iter()
            .map(Into::into)
            .collect();

        let element_type = data.dtype();
        let py = data.py();

        let scale_factor = scale_factor.unwrap_or(1.0);
        let add_offset = add_offset.unwrap_or(0.0);
        let compression = compression
            .map(|s| PyCompressionType::from_str(s))
            .transpose()?
            .unwrap_or(PyCompressionType::PforDelta2d)
            .to_omfilesrs();

        let array_meta = if element_type.is_equiv_to(&dtype::<f32>(py)) {
            let array = data.downcast::<PyArrayDyn<f32>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<f64>(py)) {
            let array = data.downcast::<PyArrayDyn<f64>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i32>(py)) {
            let array = data.downcast::<PyArrayDyn<i32>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i64>(py)) {
            let array = data.downcast::<PyArrayDyn<i64>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u32>(py)) {
            let array = data.downcast::<PyArrayDyn<u32>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u64>(py)) {
            let array = data.downcast::<PyArrayDyn<u64>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i8>(py)) {
            let array = data.downcast::<PyArrayDyn<i8>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u8>(py)) {
            let array = data.downcast::<PyArrayDyn<u8>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<i16>(py)) {
            let array = data.downcast::<PyArrayDyn<i16>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else if element_type.is_equiv_to(&dtype::<u16>(py)) {
            let array = data.downcast::<PyArrayDyn<u16>>()?.readonly();
            self.write_array_internal(array, chunks, scale_factor, add_offset, compression)
        } else {
            Err(OmFilesRsError::InvalidDataType)
        };

        let array_meta = array_meta.map_err(convert_omfilesrs_error)?;

        let offset_size = self
            .file_writer
            .write_array(array_meta, name, &children)
            .map_err(convert_omfilesrs_error)?;

        Ok(OmVariable {
            name: name.to_string(),
            offset: offset_size.offset,
            size: offset_size.size,
        })
    }
    #[pyo3(
        text_signature = "(key, value, children=None)",
        signature = (key, value, children=None)
    )]
    fn write_attribute(
        &mut self,
        key: &str,
        value: &Bound<PyAny>,
        children: Option<Vec<OmVariable>>,
    ) -> PyResult<OmVariable> {
        let children: Vec<OmOffsetSize> = children
            .unwrap_or_default()
            .iter()
            .map(Into::into)
            .collect();

        let result = if let Ok(_value) = value.extract::<String>() {
            unimplemented!("Strings are currently not implemented");
        } else if let Ok(value) = value.extract::<f64>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<f32>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<i64>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<i32>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<i16>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<i8>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<u64>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<u32>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<u16>() {
            self.store_scalar(value, key, &children)?
        } else if let Ok(value) = value.extract::<u8>() {
            self.store_scalar(value, key, &children)?
        } else {
            return Err(PyValueError::new_err(format!(
                    "Unsupported attribute type for key '{}'. Supported types are: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64",
                    key
                )));
        };
        Ok(result)
    }
}

impl OmFilePyWriter {
    fn write_array_internal<'py, T>(
        &mut self,
        data: PyReadonlyArrayDyn<'py, T>,
        chunks: Vec<u64>,
        scale_factor: f32,
        add_offset: f32,
        compression: CompressionType,
    ) -> Result<OmFileWriterArrayFinalized, OmFilesRsError>
    where
        T: Element + OmFileArrayDataType,
    {
        let dimensions = data
            .shape()
            .into_iter()
            .map(|x| *x as u64)
            .collect::<Vec<u64>>();

        let mut writer = self.file_writer.prepare_array::<T>(
            dimensions,
            chunks,
            compression,
            scale_factor,
            add_offset,
        )?;

        writer.write_data(data.as_array(), None, None)?;

        let variable_meta = writer.finalize();
        Ok(variable_meta)
    }

    fn store_scalar<T: OmFileScalarDataType + 'static>(
        &mut self,
        value: T,
        name: &str,
        children: &[OmOffsetSize],
    ) -> PyResult<OmVariable> {
        let offset_size = self
            .file_writer
            .write_scalar(value, name, children)
            .map_err(convert_omfilesrs_error)?;

        Ok(OmVariable {
            name: name.to_string(),
            offset: offset_size.offset,
            size: offset_size.size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::ArrayD, PyArrayDyn, PyArrayMethods};
    use std::fs;

    #[test]
    fn test_write_array() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test parameters
            let file_path = "test_data.om";
            let dimensions = vec![10, 20];
            let chunks = vec![5u64, 5];

            // Create test data
            let data = ArrayD::from_shape_fn(dimensions, |idx| (idx[0] + idx[1]) as f32);
            let py_array = PyArrayDyn::from_array(py, &data);

            let mut file_writer = OmFilePyWriter::new(file_path).unwrap();

            // Write data
            let result = file_writer.write_array(
                py_array.as_untyped(),
                chunks,
                None,
                None,
                None,
                None,
                None,
            );

            assert!(result.is_ok());
            assert!(fs::metadata(file_path).is_ok());

            // Clean up
            fs::remove_file(file_path).unwrap();
        });

        Ok(())
    }
}
