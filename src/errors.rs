use pyo3::PyErr;

/// Utility function to convert an OmFilesRsError to a PyRuntimeError
pub fn convert_omfilesrs_error(e: omfiles_rs::errors::OmFilesRsError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}
