use crate::{compression::PyCompressionType, errors::convert_omfilesrs_error};
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
use std::{
    collections::{HashMap, HashSet},
    fs::File,
};

enum VariableType {
    Array(OmFileWriterArrayFinalized),
    Scalar {
        write_fn: Box<
            dyn FnOnce(
                    &mut OmFileWriter<File>,
                    &[OmOffsetSize],
                ) -> Result<OmOffsetSize, OmFilesRsError>
                + 'static,
        >,
    },
}

struct VariableTree {
    nodes: HashMap<String, VariableNode>,
    pending_parents: HashMap<String, HashSet<String>>,
    root: Option<String>,
}

struct VariableNode {
    variable: VariableType,
    parent: Option<String>,
    children: Vec<String>,
}

impl VariableTree {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            pending_parents: HashMap::new(),
            root: None,
        }
    }

    fn add_node(&mut self, name: String, variable: VariableType) -> PyResult<()> {
        // Validate that the name is unique
        if self.nodes.contains_key(&name) {
            return Err(PyValueError::new_err(format!(
                "Variable '{}' already exists",
                name
            )));
        }

        // Create the new node
        let node = VariableNode {
            variable,
            children: Vec::new(),
            parent: None,
        };

        // If this node was mentioned as a child before, move it from pending to actual
        if let Some(potential_parents) = self.pending_parents.remove(&name) {
            // Note: Currently we don't know which one will be the actual parent
            // This will be resolved when set_children is called on the parent
        }

        self.nodes.insert(name.clone(), node);

        // Automatically set this as the root if it's not going to be a child
        if !self
            .pending_parents
            .values()
            .any(|parents| parents.contains(&name))
        {
            self.root = Some(name);
        }

        Ok(())
    }

    fn set_children(&mut self, parent_name: &str, children: Vec<String>) -> PyResult<()> {
        // Validate that parent exists
        if !self.nodes.contains_key(parent_name) {
            return Err(PyValueError::new_err(format!(
                "Parent variable '{}' does not exist",
                parent_name
            )));
        }

        // Validate children and collect which ones need parent updates
        let mut children_to_update = Vec::new();
        for child_name in &children {
            if let Some(node) = self.nodes.get(child_name) {
                // If child already has a parent, that's an error
                if node.parent.is_some() {
                    return Err(PyValueError::new_err(format!(
                        "Child '{}' already has a parent",
                        child_name
                    )));
                }
                children_to_update.push(child_name.clone());
            } else {
                // Child doesn't exist yet, add to pending
                self.pending_parents
                    .entry(child_name.clone())
                    .or_default()
                    .insert(parent_name.to_string());
            }
        }

        // Now we can update all the children
        for child_name in children_to_update {
            if let Some(node) = self.nodes.get_mut(&child_name) {
                node.parent = Some(parent_name.to_string());
            }
        }

        // Finally update the parent's children list
        if let Some(parent_node) = self.nodes.get_mut(parent_name) {
            parent_node.children = children;
        }

        Ok(())
    }

    fn set_root(&mut self, name: String) -> PyResult<()> {
        if !self.nodes.contains_key(&name) {
            return Err(PyValueError::new_err(format!(
                "Variable '{}' does not exist",
                name
            )));
        }
        self.root = Some(name);
        Ok(())
    }

    fn get_write_order(&self) -> Vec<String> {
        let mut order = Vec::new();
        let mut visited = HashMap::new();

        fn visit(
            name: &str,
            tree: &VariableTree,
            visited: &mut HashMap<String, bool>,
            order: &mut Vec<String>,
        ) {
            if visited.get(name).copied() == Some(true) {
                return;
            }

            visited.insert(name.to_string(), true);

            if let Some(node) = tree.nodes.get(name) {
                for child in &node.children {
                    visit(child, tree, visited, order);
                }
                order.push(name.to_string());
            }
        }

        // Start with root if specified, otherwise visit all nodes
        if let Some(root) = &self.root {
            visit(root, self, &mut visited, &mut order);
        } else {
            // Visit all nodes that haven't been visited yet
            for name in self.nodes.keys() {
                if visited.get(name).copied() != Some(true) {
                    visit(name, self, &mut visited, &mut order);
                }
            }
        }

        order
    }

    fn validate_tree(&self) -> PyResult<()> {
        // Check if there are any remaining pending parents
        if !self.pending_parents.is_empty() {
            let missing_children: Vec<_> = self.pending_parents.keys().cloned().collect();
            return Err(PyValueError::new_err(format!(
                "Some children were specified but never created: {:?}",
                missing_children
            )));
        }

        // Validate that no cycles exist
        self.check_for_cycles()?;

        Ok(())
    }

    fn check_for_cycles(&self) -> PyResult<()> {
        let mut visited = HashMap::new();
        let mut stack = HashSet::new();

        fn visit(
            name: &str,
            tree: &VariableTree,
            visited: &mut HashMap<String, bool>,
            stack: &mut HashSet<String>,
        ) -> PyResult<()> {
            if stack.contains(name) {
                return Err(PyValueError::new_err("Cycle detected in variable tree"));
            }

            if visited.get(name).copied() == Some(true) {
                return Ok(());
            }

            visited.insert(name.to_string(), true);
            stack.insert(name.to_string());

            if let Some(node) = tree.nodes.get(name) {
                for child in &node.children {
                    visit(child, tree, visited, stack)?;
                }
            }

            stack.remove(name);
            Ok(())
        }

        for name in self.nodes.keys() {
            if visited.get(name).copied() != Some(true) {
                visit(name, self, &mut visited, &mut stack)?;
            }
        }

        Ok(())
    }
}

#[pyclass(unsendable)]
pub struct OmFilePyWriter {
    file_writer: OmFileWriter<File>,
    variable_tree: VariableTree,
}

#[pymethods]
impl OmFilePyWriter {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        let file_handle = File::create(file_path)?;
        let writer = OmFileWriter::new(file_handle, 8 * 1024); // initial capacity of 8KB
        Ok(Self {
            file_writer: writer,
            variable_tree: VariableTree::new(),
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> PyResult<()> {
        self.variable_tree.validate_tree()?;

        // Get the write order based on the tree structure (bottom-up)
        let write_order = self.variable_tree.get_write_order();
        let mut offset_sizes: HashMap<String, OmOffsetSize> = HashMap::new();
        let mut nodes = std::mem::take(&mut self.variable_tree.nodes);

        // Process all nodes in order (bottom-up)
        for name in &write_order {
            if let Some(node) = nodes.remove(name) {
                // Get the OmOffsetSize for each child
                let children_offsets: Vec<OmOffsetSize> = node
                    .children
                    .iter()
                    .map(|child_name| {
                        offset_sizes
                            .get(child_name)
                            .ok_or_else(|| {
                                PyValueError::new_err(format!(
                                    "Child '{}' has not been written yet",
                                    child_name
                                ))
                            })
                            .cloned()
                    })
                    .collect::<PyResult<Vec<_>>>()?;

                // Write the variable (either scalar or array) and store its offset
                let offset_size = match node.variable {
                    VariableType::Scalar { write_fn } => {
                        write_fn(&mut self.file_writer, &children_offsets)
                            .map_err(convert_omfilesrs_error)?
                    }
                    VariableType::Array(array_meta) => self
                        .file_writer
                        .write_array(&array_meta, name, &children_offsets)
                        .map_err(convert_omfilesrs_error)?,
                };

                offset_sizes.insert(name.clone(), offset_size);
            }
        }

        // Write the trailer using the root
        if let Some(root_name) = &self.variable_tree.root {
            if let Some(root_offset_size) = offset_sizes.get(root_name) {
                self.file_writer
                    .write_trailer(root_offset_size.clone())
                    .map_err(convert_omfilesrs_error)?;
            }
        }

        Ok(())
    }

    #[pyo3(
            text_signature = "(data, chunks, /, *, scale_factor=1.0, add_offset=0.0, compression='pfor_delta_2d', name='data', children=[])",
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
        children: Option<Vec<String>>,
    ) -> PyResult<()> {
        let name = name.unwrap_or("data");
        let children = children.unwrap_or_default();

        self.validate_children(&children)?;
        self.validate_unique_name(name)?;

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
        }
        .map_err(convert_omfilesrs_error)?;

        self.variable_tree
            .add_node(name.to_string(), VariableType::Array(array_meta))?;

        if !children.is_empty() {
            self.variable_tree.set_children(name, children)?;
        }

        Ok(())
    }

    #[pyo3(
        text_signature = "(name, value, /, *, children=None)",
        signature = (name, value, children=None)
    )]
    fn write_attribute(
        &mut self,
        name: &str,
        value: &Bound<PyAny>,
        children: Option<Vec<String>>,
    ) -> PyResult<()> {
        let children = children.unwrap_or_default();

        if let Ok(_value) = value.extract::<String>() {
            unimplemented!("Strings are currently not implemented");
            // self.file_writer
            //     .write_scalar(value, key, children)
            //     .map_err(convert_omfilesrs_error)?;
        } else if let Ok(value) = value.extract::<f64>() {
            self.store_scalar(name.to_string(), value, children)?;
            // self.file_writer
            //     .write_scalar(value, name, &[])
            //     .map_err(convert_omfilesrs_error)?;
        } else if let Ok(value) = value.extract::<f32>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<i64>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<i32>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<i16>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<i8>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<u64>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<u32>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<u16>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else if let Ok(value) = value.extract::<u8>() {
            self.store_scalar(name.to_string(), value, children)?;
        } else {
            return Err(PyValueError::new_err(format!(
                    "Unsupported attribute type for key '{}'. Supported types are: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64",
                    name
                )));
        };
        Ok(())
    }
}

impl OmFilePyWriter {
    /// Validates that all specified children exist in the writer's variables
    fn validate_children(&self, children: &[String]) -> PyResult<()> {
        for child in children {
            if !self.variable_tree.nodes.contains_key(child) {
                return Err(PyValueError::new_err(format!(
                    "Child variable '{}' does not exist",
                    child
                )));
            }
        }
        Ok(())
    }

    /// Validates that a variable name is not already in use
    fn validate_unique_name(&self, name: &str) -> PyResult<()> {
        if self.variable_tree.nodes.contains_key(name) {
            return Err(PyValueError::new_err(format!(
                "Variable '{}' already exists",
                name
            )));
        }
        Ok(())
    }

    fn store_scalar<T: OmFileScalarDataType + 'static>(
        &mut self,
        name: String,
        value: T,
        children: Vec<String>,
    ) -> PyResult<()> {
        self.validate_children(&children)?;
        self.validate_unique_name(&name)?;

        let name_for_closure = name.clone();

        let write_fn = Box::new(
            move |writer: &mut OmFileWriter<File>, children_offsets: &[OmOffsetSize]| {
                writer.write_scalar(value, &name_for_closure, children_offsets)
            },
        );

        self.variable_tree
            .add_node(name.clone(), VariableType::Scalar { write_fn })?;

        if !children.is_empty() {
            self.variable_tree.set_children(&name, children)?;
        }
        Ok(())
    }

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
        // let variable = self
        //     .file_writer
        //     .write_array(variable_meta, name, &[])
        //     .map_err(convert_omfilesrs_error)?;
        // self.file_writer
        //     .write_trailer(variable)
        //     .map_err(convert_omfilesrs_error)?;

        // Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use numpy::{ndarray::ArrayD, PyArrayDyn, PyArrayMethods};
    use omfiles_rs::{
        backend::mmapfile::{MmapFile, Mode},
        io::reader::OmFileReader,
    };
    use std::{fs, sync::Arc};

    #[test]
    fn test_write_and_read_array() -> Result<(), Box<dyn std::error::Error>> {
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

            // Write data using context manager
            {
                let writer = OmFilePyWriter::new(file_path).unwrap();
                let mut writer_guard = writer.into_py(py);

                // Enter context
                let writer = writer_guard
                    .call_method0(py, "__enter__")
                    .expect("Failed to enter context");

                // Write data and set as root
                writer
                    .call_method(
                        py,
                        "write_array",
                        (
                            py_array.as_untyped(),
                            chunks,
                            None::<f32>,         // scale_factor
                            None::<f32>,         // add_offset
                            None::<String>,      // compression
                            Some("root"),        // name
                            None::<Vec<String>>, // children
                        ),
                        None,
                    )
                    .expect("Failed to write array");

                // Exit context
                writer_guard
                    .call_method(py, "__exit__", (py.None(), py.None(), py.None()), None)
                    .expect("Failed to exit context");
            }

            // Verify file exists
            assert!(fs::metadata(file_path).is_ok());

            // Read and verify data
            {
                let file_for_reading = File::open(file_path).expect("Failed to open file");
                let read_backend =
                    MmapFile::new(file_for_reading, Mode::ReadOnly).expect("Failed to mmap file");
                let reader =
                    OmFileReader::new(Arc::new(read_backend)).expect("Failed to create reader");
                let result = reader
                    .read::<f32>(&[0..10, 0..20], None, None)
                    .expect("Failed to read");
                assert_eq!(result, data);
            }

            // Clean up
            fs::remove_file(file_path).unwrap();
        });

        Ok(())
    }
}
