use pyo3::prelude::*;
use std::ops::Range;

/// A simplified numpy-like array indexing implementation.
/// Supports integer, slice, and newaxis indexing.
/// Currently does not support ellipsis, boolean, or integer array indexing.
#[derive(Debug)]
pub enum IndexType {
    Int(i64),
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    NewAxis,
    // Ellipsis,
    // Boolean(Array<bool, Ix1>),
    // IntArray(Array<i64, Ix1>),
}

#[derive(Debug)]
pub struct ArrayIndex(pub Vec<IndexType>);

impl<'py> FromPyObject<'py> for ArrayIndex {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(tuple) = ob.downcast::<pyo3::types::PyTuple>() {
            let mut indices = Vec::new();

            for item in tuple.iter() {
                let idx = if item.is_instance_of::<pyo3::types::PySlice>() {
                    let slice = item.downcast::<pyo3::types::PySlice>()?;
                    let start = slice.getattr("start")?.extract()?;
                    let stop = slice.getattr("stop")?.extract()?;
                    let step = slice.getattr("step")?.extract()?;
                    IndexType::Slice { start, stop, step }
                // } else if item.is_instance_of::<pyo3::types::PyEllipsis>() {
                //     IndexType::Ellipsis
                } else if item.is_none() {
                    IndexType::NewAxis
                // } else if let Ok(array) = item.extract::<ArrayView<bool, Ix1>>() {
                //     IndexType::Boolean(array.to_owned())
                // } else if let Ok(array) = item.extract::<ArrayView<i64, Ix1>>() {
                //     IndexType::IntArray(array.to_owned())
                } else {
                    IndexType::Int(item.extract()?)
                };
                indices.push(idx);
            }

            Ok(ArrayIndex(indices))
        } else {
            let idx = if ob.is_instance_of::<pyo3::types::PySlice>() {
                let slice = ob.downcast::<pyo3::types::PySlice>()?;
                let start = slice.getattr("start")?.extract()?;
                let stop = slice.getattr("stop")?.extract()?;
                let step = slice.getattr("step")?.extract()?;
                IndexType::Slice { start, stop, step }
            } else if ob.is_none() {
                IndexType::NewAxis
            } else {
                IndexType::Int(ob.extract()?)
            };
            Ok(ArrayIndex(vec![idx]))
        }
    }
}

impl ArrayIndex {
    pub fn to_read_range(&self, shape: &Vec<u64>) -> PyResult<Vec<Range<u64>>> {
        // Input validation
        if self.0.len() > shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices for array",
            ));
        }

        if self.0.len() < shape.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Not enough indices for array",
            ));
        }
        self.validate_against_shape(shape)?;

        let mut ranges = Vec::new();
        let mut shape_idx = 0;

        for idx in &self.0 {
            match idx {
                IndexType::Int(i) => {
                    let i = if *i < 0 {
                        (shape[shape_idx] as i64 + *i) as u64
                    } else {
                        *i as u64
                    };
                    ranges.push(Range {
                        start: i,
                        end: i + 1,
                    });
                    shape_idx += 1;
                }
                IndexType::Slice { start, stop, step } => {
                    let dim_size = shape[shape_idx];

                    // Handle start
                    let start_idx = match start {
                        Some(s) => {
                            if *s < 0 {
                                ((dim_size as i64) + s) as u64
                            } else {
                                *s as u64
                            }
                        }
                        None => 0,
                    };

                    // Handle stop
                    let stop_idx = match stop {
                        Some(s) => {
                            if *s < 0 {
                                ((dim_size as i64) + s) as u64
                            } else {
                                *s as u64
                            }
                        }
                        None => dim_size,
                    };

                    // Validate step (already checked in validate_against_shape)
                    if let Some(step) = step {
                        if *step != 1 {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Slice step must be 1",
                            ));
                        }
                    }

                    // Create range
                    ranges.push(Range {
                        start: start_idx,
                        end: stop_idx,
                    });
                    shape_idx += 1;
                }
                IndexType::NewAxis => {
                    // NewAxis returns the full dimension
                    ranges.push(Range {
                        start: 0,
                        end: shape[shape_idx],
                    });
                }
            }
        }

        // Handle remaining dimensions if any
        while shape_idx < shape.len() {
            ranges.push(Range {
                start: 0,
                end: shape[shape_idx],
            });
            shape_idx += 1;
        }

        Ok(ranges)
    }

    pub fn validate_against_shape(&self, shape: &Vec<u64>) -> PyResult<()> {
        for (idx, &dim_size) in self.0.iter().zip(shape.iter()) {
            match idx {
                IndexType::Int(i) => {
                    let i = *i as u64;
                    if i >= dim_size {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                            "Index {} is out of bounds for axis with size {}",
                            i, dim_size
                        )));
                    }
                }
                IndexType::Slice { start, stop, step } => {
                    if let Some(start) = start {
                        let start = *start as u64;
                        if start > dim_size {
                            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                                "Index {} is out of bounds for axis with size {}",
                                start, dim_size
                            )));
                        }
                    }
                    if let Some(stop) = stop {
                        let stop = *stop as u64;
                        if stop > dim_size {
                            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                                "Index {} is out of bounds for axis with size {}",
                                stop, dim_size
                            )));
                        }
                    }
                    if let Some(step) = step {
                        if *step != 1 {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Slice step must be 1",
                            ));
                        }
                    }
                }
                IndexType::NewAxis => continue,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PySlice;

    #[test]
    fn test_numpy_indexing() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Test basic slicing
            let slice = PySlice::new(py, 1, 5, 1);
            let single_slice_tuple = pyo3::types::PyTuple::new(py, &[slice.as_ref()]).unwrap();
            let slice_index = ArrayIndex::extract_bound(single_slice_tuple.as_ref()).unwrap();
            match &slice_index.0[0] {
                IndexType::Slice { start, stop, step } => {
                    assert_eq!(*start, Some(1));
                    assert_eq!(*stop, Some(5));
                    assert_eq!(*step, Some(1));
                }
                _ => panic!("Expected Slice type"),
            }

            // Test integer indexing
            let int_value = 42i64.into_pyobject(py).unwrap();
            let single_int_tuple = pyo3::types::PyTuple::new(py, &[int_value.as_ref()]).unwrap();
            let int_index = ArrayIndex::extract_bound(single_int_tuple.as_ref()).unwrap();
            match int_index.0[0] {
                IndexType::Int(val) => assert_eq!(val, 42),
                _ => panic!("Expected Int type"),
            }

            // Test None (NewAxis)
            let none_value = py.None();
            let single_none_tuple = pyo3::types::PyTuple::new(py, &[none_value]).unwrap();
            let none_index = ArrayIndex::extract_bound(single_none_tuple.as_ref()).unwrap();
            match none_index.0[0] {
                IndexType::NewAxis => (),
                _ => panic!("Expected NewAxis type"),
            }

            // Test combination of different types
            let mixed_tuple = pyo3::types::PyTuple::new(
                py,
                &[
                    slice.as_ref(),                            // slice
                    42i64.into_pyobject(py).unwrap().as_ref(), // integer
                    py.None().bind(py),                        // NewAxis
                ],
            )
            .unwrap();
            let mixed_index = ArrayIndex::extract_bound(mixed_tuple.as_ref()).unwrap();

            // Verify the types in order
            match &mixed_index.0[0] {
                IndexType::Slice { start, stop, step } => {
                    assert_eq!(*start, Some(1));
                    assert_eq!(*stop, Some(5));
                    assert_eq!(*step, Some(1));
                }
                _ => panic!("Expected Slice type"),
            }

            match mixed_index.0[1] {
                IndexType::Int(val) => assert_eq!(val, 42),
                _ => panic!("Expected Int type"),
            }

            match mixed_index.0[2] {
                IndexType::NewAxis => (),
                _ => panic!("Expected NewAxis type"),
            }

            // Test slice with None values (open-ended slices)
            let open_slice = PySlice::full(py);
            let open_slice_tuple = pyo3::types::PyTuple::new(py, &[open_slice.as_ref()]).unwrap();
            let open_slice_index = ArrayIndex::extract_bound(open_slice_tuple.as_ref()).unwrap();
            match &open_slice_index.0[0] {
                IndexType::Slice { start, stop, step } => {
                    assert_eq!(*start, None);
                    assert_eq!(*stop, None);
                    assert_eq!(*step, None);
                }
                _ => panic!("Expected Slice type"),
            }
        });
    }

    #[test]
    #[should_panic]
    fn test_invalid_input() {
        Python::with_gil(|py| {
            let invalid_value = "not_an_index"
                .into_pyobject(py)
                .expect("Failed to create object");
            let invalid_tuple = pyo3::types::PyTuple::new(py, &[invalid_value.as_ref()])
                .expect("Failed to create tuple");
            let _should_fail = ArrayIndex::extract_bound(invalid_tuple.as_ref()).unwrap();
        });
    }
}
