use omfiles_rs::io::writer::OmOffsetSize;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub(crate) struct OmVariable {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub offset: u64,
    #[pyo3(get)]
    pub size: u64,
}

#[pymethods]
impl OmVariable {
    fn __repr__(&self) -> String {
        format!(
            "OmVariable(name='{}', offset={}, size={})",
            self.name, self.offset, self.size
        )
    }
}

impl Into<OmOffsetSize> for &OmVariable {
    fn into(self) -> OmOffsetSize {
        OmOffsetSize {
            offset: self.offset,
            size: self.size,
        }
    }
}

impl Into<OmOffsetSize> for OmVariable {
    fn into(self) -> OmOffsetSize {
        OmOffsetSize {
            offset: self.offset,
            size: self.size,
        }
    }
}
