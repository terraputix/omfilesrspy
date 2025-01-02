use omfiles_rs::backend::backends::OmFileReaderBackend;
use pyo3::prelude::*;
use pyo3::Python;
use std::error::Error;

#[pyclass]
pub struct FsSpecBackend {
    py_file: PyObject,
    #[pyo3(get)]
    file_size: u64,
}

#[pymethods]
impl FsSpecBackend {
    #[new]
    pub fn new(open_file: PyObject) -> PyResult<Self> {
        let size = Python::with_gil(|py| -> PyResult<u64> {
            let fs = open_file.bind(py).getattr("fs")?;
            let path = open_file.bind(py).getattr("path")?;
            let info = fs.call_method1("info", (path,))?;
            let size = info.get_item("size")?.extract::<u64>()?;
            Ok(size)
        })?;

        Ok(Self {
            py_file: open_file.into(),
            file_size: size,
        })
    }
}

impl OmFileReaderBackend for FsSpecBackend {
    fn count(&self) -> usize {
        self.file_size as usize
    }

    fn needs_prefetch(&self) -> bool {
        false
    }

    fn prefetch_data(&self, _offset: usize, _count: usize) {
        // No-op for now
    }

    fn pre_read(
        &self,
        _offset: usize,
        _count: usize,
    ) -> Result<(), omfiles_rs::errors::OmFilesRsError> {
        Ok(())
    }

    fn get_bytes_owned(
        &self,
        offset: u64,
        count: u64,
    ) -> Result<Vec<u8>, omfiles_rs::errors::OmFilesRsError> {
        let bytes = Python::with_gil(|py| -> Result<Vec<u8>, Box<dyn Error>> {
            // Seek to offset
            self.py_file.call_method1(py, "seek", (offset,))?;

            // Read count bytes
            let bytes = self.py_file.call_method1(py, "read", (count,))?;
            let py_bytes = bytes.extract::<Vec<u8>>(py)?;

            Ok(py_bytes)
        })
        // FIXME: error type
        .map_err(|e| omfiles_rs::errors::OmFilesRsError::DecoderError(e.to_string()))?;

        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_fsspec_backend() -> Result<(), Box<dyn Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> Result<(), Box<dyn Error>> {
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method1("filesystem", ("file",))?;
            let file_path = "test_files/read_test.om";
            let open_file = fs.call_method1("open", (file_path,))?;

            // Create FsSpecBackend
            let backend = FsSpecBackend::new(open_file.into())?;
            assert_eq!(backend.file_size, 144);

            let bytes = backend.get_bytes_owned(0, 44)?;
            assert_eq!(
                &bytes,
                &[
                    79, 77, 3, 0, 4, 130, 0, 2, 3, 34, 0, 4, 194, 2, 10, 4, 178, 0, 12, 4, 242, 0,
                    14, 197, 17, 20, 194, 2, 22, 194, 2, 24, 3, 3, 228, 200, 109, 1, 0, 0, 20, 0,
                    4, 0
                ]
            );

            Ok(())
        })?;

        Ok(())
    }
}
