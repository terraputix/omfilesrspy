use pyo3::PyErr;

/// Utility function to convert an OmFilesRsError to a PyRuntimeError
pub fn convert_omfilesrs_error(e: crate::errors::OmFilesRsError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}
