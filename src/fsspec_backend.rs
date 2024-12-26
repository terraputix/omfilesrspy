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

/// Different types of caches that can be used with the caching filesystem
/// Compare: https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.cached.CachingFileSystem
pub enum CacheType {
    /// Caches whole remote files on first access
    SimpleCacheFileSystem,
    /// Chunkwise local storage in a sparse file
    CachingFileSystem,
    /// Caches whole remote files on first access, similar to CachingFileSystem
    /// but without requiring a sparse file
    WholeFileCacheFileSystem,
}

impl CacheType {
    pub fn to_str(&self) -> &str {
        match self {
            CacheType::SimpleCacheFileSystem => "SimpleCacheFileSystem",
            CacheType::CachingFileSystem => "CachingFileSystem",
            CacheType::WholeFileCacheFileSystem => "WholeFileCacheFileSystem",
        }
    }
}

pub struct CacheSettings {
    pub cache_dir: PathBuf,
    pub expiry_time: Option<u64>,
    pub cache_type: CacheType,
}

impl FsSpecBackend {
    pub fn new(
        path: &str,
        protocol: Option<&str>,
        kwargs: Option<&Bound<'_, PyDict>>,
        cache_settings: Option<CacheSettings>,
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
            // Determine protocol and path
            let protocol = protocol.unwrap_or("file");
            let final_path = absolute_path.to_string_lossy().to_string();

            // Create filesystem without cache
            let fsspec = py.import("fsspec")?;
            let fs = fsspec.call_method("filesystem", (protocol,), kwargs)?;

            // Potentially overwrite filesystem with caching filesystem
            let fs = if let Some(cache_settings) = cache_settings {
                let cached_fs = py
                    .import("fsspec.implementations.cached")?
                    .getattr(cache_settings.cache_type.to_str())?;

                // Create cache filesystem options
                let cache_options = PyDict::new(py);
                cache_options.set_item("fs", fs)?;
                cache_options
                    .set_item("cache_storage", cache_settings.cache_dir.to_str().unwrap())?;
                if let Some(expiry) = cache_settings.expiry_time {
                    cache_options.set_item("expiry_time", expiry)?;
                }
                // Return the caching filesystem
                cached_fs.call((), Some(&cache_options))?
            } else {
                fs
            };

            // "open" the file and get the file size
            let py_file = fs.call_method("open", (final_path.clone(),), None)?;
            let info = fs.call_method("info", (final_path,), None)?;
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

        let file = FsSpecBackend::new("test_files/read_test.om", Some("file"), None, None)?;
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
                None,
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
                None,
            )?;

            // Create OmFileReader with the FsSpecBackend
            let reader =
                OmFileReader2::new(Arc::new(backend)).map_err(|e| Box::new(e) as Box<dyn Error>)?;

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

    #[test]
    fn test_s3_reader_with_cache() -> Result<(), Box<dyn Error>> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> Result<(), Box<dyn Error>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("anon", true)?;

            let cache_settings = CacheSettings {
                cache_dir: PathBuf::from("/tmp/fsspec_cache"),
                expiry_time: None, // Use Python default expiry time
                cache_type: CacheType::SimpleCacheFileSystem,
            };

            // Initialize the FsSpecBackend with cache
            let backend = FsSpecBackend::new(
                "openmeteo/data/dwd_icon_d2/temperature_2m/chunk_3960.om",
                Some("s3"),
                Some(&kwargs),
                Some(cache_settings),
            )?;

            // Create OmFileReader with the FsSpecBackend
            let reader =
                OmFileReader2::new(Arc::new(backend)).map_err(|e| Box::new(e) as Box<dyn Error>)?;

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
