use omfiles_rs::backend::backends::OmFileReaderBackend;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;
use std::ops::Range;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_fsspec_backend_new() -> Result<(), Box<dyn Error>> {
        pyo3::prepare_freethreaded_python();

        let file = FsSpecBackend::new("test_files/read_test.om", Some("file"), None)?;

        assert_eq!(file.file_size, 144);

        Ok(())
    }
}
