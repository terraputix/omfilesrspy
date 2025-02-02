use omfiles_rs::{
    errors::OmFilesRsError,
    io::writer::{OmFileWriter, OmFileWriterArrayFinalized, OmOffsetSize},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
};

use crate::errors::convert_omfilesrs_error;

pub enum VariableType {
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

pub struct VariableTree {
    pub nodes: HashMap<String, VariableNode>,
    pending_parents: HashMap<String, HashSet<String>>,
    root: Option<String>,
}

pub struct VariableNode {
    variable: VariableType,
    parent: Option<String>,
    children: Vec<String>,
}

impl VariableTree {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            pending_parents: HashMap::new(),
            root: None,
        }
    }

    pub fn write_variables(
        &mut self,
        writer: &mut OmFileWriter<File>,
    ) -> PyResult<Option<OmOffsetSize>> {
        self.validate_tree()?;

        // Get the write order based on the tree structure (bottom-up)
        let write_order = self.get_write_order();
        let mut offset_sizes: HashMap<String, OmOffsetSize> = HashMap::new();
        let mut nodes = std::mem::take(&mut self.nodes);

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
                        write_fn(writer, &children_offsets).map_err(convert_omfilesrs_error)?
                    }
                    VariableType::Array(array_meta) => writer
                        .write_array(&array_meta, name, &children_offsets)
                        .map_err(convert_omfilesrs_error)?,
                };

                offset_sizes.insert(name.clone(), offset_size);
            }
        }

        // Return the root's offset size if it exists
        Ok(self
            .root
            .as_ref()
            .and_then(|root_name| offset_sizes.get(root_name).cloned()))
    }

    pub fn add_node(&mut self, name: String, variable: VariableType) -> PyResult<()> {
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
        if let Some(_potential_parents) = self.pending_parents.remove(&name) {
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

    pub fn set_children(&mut self, parent_name: &str, children: Vec<String>) -> PyResult<()> {
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

    pub fn get_write_order(&self) -> Vec<String> {
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

    pub fn validate_tree(&self) -> PyResult<()> {
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

    pub fn check_for_cycles(&self) -> PyResult<()> {
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
