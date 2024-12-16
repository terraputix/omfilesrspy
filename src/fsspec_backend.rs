use omfiles_rs::backend::backends::OmFileReaderBackend;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;
use std::error::Error;
use std::path::PathBuf;

pub struct FsSpecBackend {
    py_file: PyObject,
    file_size: u64,
}

impl FsSpecBackend {
    pub fn new(
        path: &str,
        protocol: Option<&str>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Convert path to absolute path if it's a local file
            let path_buf = PathBuf::from(path);
            let absolute_path = if protocol.is_none() || protocol == Some("file") {
                path_buf.canonicalize().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to resolve path: {}",
                        e
                    ))
                })?
            } else {
                path_buf
            };

            // Use fsspec for string paths
            let fsspec = py.import("fsspec")?;

            // Determine protocol and path
            let (protocol, final_path) = if let Some(proto) = protocol {
                (proto, absolute_path.to_string_lossy().to_string())
            } else {
                ("file", absolute_path.to_string_lossy().to_string())
            };

            // Create filesystem with specified protocol
            let fs = if let Some(kwargs) = kwargs {
                fsspec.call_method("filesystem", (protocol,), Some(kwargs))?
            } else {
                fsspec.call_method("filesystem", (protocol,), None)?
            };

            let py_file = fs.call_method("open", (final_path,), None)?;
            // Get file size
            let info = fs.call_method("info", (path,), None)?;
            let size = info.get_item("size")?.extract::<u64>()?;

            Ok(Self {
                py_file: py_file.into(),
                file_size: size,
            })
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
    use omfiles_rs::io::reader2::OmFileReader2;
    use std::{error::Error, ops::Range, sync::Arc};

    #[test]
    fn test_fsspec_backend() -> Result<(), Box<dyn Error>> {
        pyo3::prepare_freethreaded_python();

        let file = FsSpecBackend::new("test_files/read_test.om", Some("file"), None)?;
        assert_eq!(file.file_size, 144);

        let bytes = file.get_bytes_owned(0, 44)?;
        assert_eq!(
            &bytes,
            &[
                79, 77, 3, 0, 4, 130, 0, 2, 3, 34, 0, 4, 194, 2, 10, 4, 178, 0, 12, 4, 242, 0, 14,
                197, 17, 20, 194, 2, 22, 194, 2, 24, 3, 3, 228, 200, 109, 1, 0, 0, 20, 0, 4, 0
            ]
        );

        Ok(())
    }

    #[test]
    fn test_fsspec_s3() -> Result<(), Box<dyn Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> Result<(), Box<dyn Error>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("anon", true)?;

            let file = FsSpecBackend::new(
                "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3980.om",
                Some("s3"),
                Some(&kwargs),
            )?;

            assert_eq!(file.file_size, 34479300);

            let bytes = file.get_bytes_owned(0, 44)?;
            assert_eq!(
                &bytes,
                &[
                    79, 77, 2, 0, 0, 0, 160, 65, 150, 212, 13, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0,
                    0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0
                ]
            );

            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_s3_reader() -> Result<(), Box<dyn Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> Result<(), Box<dyn Error>> {
            // Create S3 filesystem with anonymous access
            let kwargs = PyDict::new(py);
            kwargs.set_item("anon", true)?;

            // Initialize the FsSpecBackend
            let backend = FsSpecBackend::new(
                "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om",
                Some("s3"),
                Some(&kwargs),
            )?;

            // Create OmFileReader with the FsSpecBackend
            let reader = OmFileReader2::new(Arc::new(backend), 256)
                .map_err(|e| Box::new(e) as Box<dyn Error>)?;

            // Read a small section of data
            let ranges = vec![
                Range {
                    start: 57812,
                    end: 57813,
                },
                Range { start: 0, end: 100 },
            ];

            let data = reader
                .read_simple(&ranges, None, None)
                .map_err(|e| Box::new(e) as Box<dyn Error>)?;

            // Verify we got some data
            assert_eq!(
                &data[..10],
                &[18.0, 17.7, 17.65, 17.45, 17.15, 17.6, 18.7, 20.75, 21.7, 22.65]
            );

            Ok(())
        })?;

        Ok(())
    }
}
