use numpy::{
    dtype, Element, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArray, PyUntypedArrayMethods,
};
use omfiles_rs::{
    core::compression::CompressionType, core::data_types::OmFileArrayDataType,
    io::writer::OmFileWriter,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fs::File;

use crate::errors::convert_omfilesrs_error;

#[derive(Clone)]
pub enum PyCompressionType {
    PforDelta2dInt16,
    FpxXor2d,
    PforDelta2d,
    PforDelta2dInt16Logarithmic,
}

impl PyCompressionType {
    fn to_omfilesrs(&self) -> CompressionType {
        match self {
            PyCompressionType::PforDelta2dInt16 => CompressionType::PforDelta2dInt16,
            PyCompressionType::FpxXor2d => CompressionType::FpxXor2d,
            PyCompressionType::PforDelta2d => CompressionType::PforDelta2d,
            PyCompressionType::PforDelta2dInt16Logarithmic => {
                CompressionType::PforDelta2dInt16Logarithmic
            }
        }
    }

    fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "pfor_delta_2d_int16" => Ok(PyCompressionType::PforDelta2dInt16),
            "fpx_xor_2d" => Ok(PyCompressionType::FpxXor2d),
            "pfor_delta_2d" => Ok(PyCompressionType::PforDelta2d),
            "pfor_delta_2d_int16_logarithmic" => Ok(PyCompressionType::PforDelta2dInt16Logarithmic),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported compression type: {}",
                s
            ))),
        }
    }
}

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
            text_signature = "(data, chunks, /, *, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d')",
            signature = (data, chunks, scale_factor=None, add_offset=None, compression=None)
        )]
    fn write_array(
        &mut self,
        data: &Bound<'_, PyUntypedArray>,
        chunks: Vec<u64>,
        scale_factor: Option<f32>,
        add_offset: Option<f32>,
        compression: Option<&str>,
    ) -> PyResult<()> {
        let element_type = data.dtype();
        let py = data.py();

        let scale_factor = scale_factor.unwrap_or(1.0);
        let add_offset = add_offset.unwrap_or(0.0);
        let compression = compression
            .map(|s| PyCompressionType::from_str(s))
            .transpose()?
            .unwrap_or(PyCompressionType::PforDelta2d)
            .to_omfilesrs();

        if element_type.is_equiv_to(&dtype::<f32>(py)) {
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
            Err(PyValueError::new_err(format!(
                "Unsupported data type: {:?}",
                element_type
            )))
        }
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
    ) -> PyResult<()>
    where
        T: Element + OmFileArrayDataType,
    {
        let dimensions = data
            .shape()
            .into_iter()
            .map(|x| *x as u64)
            .collect::<Vec<u64>>();

        let mut writer = self
            .file_writer
            .prepare_array::<T>(dimensions, chunks, compression, scale_factor, add_offset)
            .map_err(convert_omfilesrs_error)?;

        writer
            .write_data(data.as_array(), None, None)
            .map_err(convert_omfilesrs_error)?;

        let variable_meta = writer.finalize();
        let variable = self
            .file_writer
            .write_array(variable_meta, "data", &[])
            .map_err(convert_omfilesrs_error)?;
        self.file_writer
            .write_trailer(variable)
            .map_err(convert_omfilesrs_error)?;

        Ok(())
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
            let result = file_writer.write_array(py_array.as_untyped(), chunks, None, None, None);

            assert!(result.is_ok());
            assert!(fs::metadata(file_path).is_ok());

            // Clean up
            fs::remove_file(file_path).unwrap();
        });

        Ok(())
    }
}
