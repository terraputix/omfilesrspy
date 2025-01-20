use numpy::{Element, PyReadonlyArrayDyn};
use omfiles_rs::{
    core::compression::CompressionType, core::data_types::OmFileArrayDataType,
    io::writer::OmFileWriter,
};
use pyo3::prelude::*;
use std::fs::File;

use crate::errors::convert_omfilesrs_error;

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

    #[pyo3(text_signature = "(data, chunks, scale_factor, add_offset, compression='p4nzdec256')")]
    fn write_array_f32(
        &mut self,
        data: PyReadonlyArrayDyn<f32>,
        chunks: Vec<u64>,
        scale_factor: f32,
        add_offset: f32,
        compression: PyCompressionType,
    ) -> PyResult<()> {
        self.write_array_internal(
            data,
            chunks,
            scale_factor,
            add_offset,
            compression.to_omfilesrs(),
        )
    }

    #[pyo3(text_signature = "(data, chunks, scale_factor, add_offset, compression='p4nzdec256')")]
    fn write_array_f64(
        &mut self,
        data: PyReadonlyArrayDyn<f64>,
        chunks: Vec<u64>,
        scale_factor: f32,
        add_offset: f32,
        compression: PyCompressionType,
    ) -> PyResult<()> {
        self.write_array_internal(
            data,
            chunks,
            scale_factor,
            add_offset,
            compression.to_omfilesrs(),
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub enum PyCompressionType {
    P4nzdec256,
    Fpxdec32,
    P4nzdec256logarithmic,
}

impl PyCompressionType {
    fn to_omfilesrs(&self) -> CompressionType {
        match self {
            PyCompressionType::P4nzdec256 => CompressionType::P4nzdec256,
            PyCompressionType::Fpxdec32 => CompressionType::Fpxdec32,
            PyCompressionType::P4nzdec256logarithmic => CompressionType::P4nzdec256logarithmic,
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
        T: Copy + Send + Sync + Element + OmFileArrayDataType,
    {
        let shape = data.getattr("shape")?.extract::<Vec<u64>>()?;
        let dimensions = shape;

        let mut writer = self
            .file_writer
            .prepare_array::<T>(dimensions, chunks, compression, scale_factor, add_offset)
            .map_err(convert_omfilesrs_error)?;

        let array = data.as_slice().expect("No array found behind `data`");

        writer
            .write_data(array, None, None, None)
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
            let scale_factor = 1.0;
            let add_offset = 0.0;

            // Create test data
            let data = ArrayD::from_shape_fn(dimensions, |idx| (idx[0] + idx[1]) as f32);
            let py_array = PyArrayDyn::from_array(py, &data);

            let mut file_writer = OmFilePyWriter::new(file_path).unwrap();

            // Write data
            let result = file_writer.write_array_f32(
                py_array.readonly(),
                chunks,
                scale_factor,
                add_offset,
                PyCompressionType::P4nzdec256,
            );

            assert!(result.is_ok());
            assert!(fs::metadata(file_path).is_ok());

            // Clean up
            fs::remove_file(file_path).unwrap();
        });

        Ok(())
    }
}
