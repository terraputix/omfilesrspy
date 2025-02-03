#![allow(non_snake_case)]
use crate::backend::backends::OmFileReaderBackend;
use crate::backend::mmapfile::{MmapFile, Mode};
use crate::core::c_defaults::{c_error_string, create_uninit_decoder};
use crate::core::compression::CompressionType;
use crate::core::data_types::{DataType, OmFileArrayDataType, OmFileScalarDataType};
use crate::errors::OmFilesRsError;
use ndarray::ArrayD;
use num_traits::Zero;
use om_file_format_sys::{
    om_decoder_init, om_decoder_read_buffer_size, om_header_size, om_header_type, om_trailer_read,
    om_trailer_size, om_variable_get_add_offset, om_variable_get_children,
    om_variable_get_children_count, om_variable_get_chunks, om_variable_get_compression,
    om_variable_get_dimensions, om_variable_get_name, om_variable_get_scalar,
    om_variable_get_scale_factor, om_variable_get_type, om_variable_init, OmError_t_ERROR_OK,
    OmHeaderType_t_OM_HEADER_INVALID, OmHeaderType_t_OM_HEADER_LEGACY,
    OmHeaderType_t_OM_HEADER_READ_TRAILER, OmVariable_t,
};
use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::os::raw::c_void;
use std::sync::Arc;

use super::writer::OmOffsetSize;

pub struct OmFileReader<Backend: OmFileReaderBackend> {
    offset_size: Option<OmOffsetSize>,
    /// The backend that provides data via the get_bytes method
    pub backend: Arc<Backend>,
    /// Holds the data where the meta information of the variable is stored, is not supposed to go out of scope
    /// Here the LUT and additional attributes of the variable need to be stored.
    pub variable_data: Vec<u8>,
    /// Opaque pointer to the variable defined by header/trailer
    pub variable: *const OmVariable_t,
}

impl<Backend: OmFileReaderBackend> OmFileReader<Backend> {
    #[allow(non_upper_case_globals)]
    pub fn new(backend: Arc<Backend>) -> Result<Self, OmFilesRsError> {
        let header_size = unsafe { om_header_size() } as u64;
        let owned_data: Result<Vec<u8>, OmFilesRsError> = backend.get_bytes_owned(0, header_size);
        let header_data = match owned_data {
            Ok(data) => data,
            Err(error) => backend
                .forward_unimplemented_error(error, || backend.get_bytes(0, header_size))?
                .to_vec(),
        };

        let header_type = unsafe { om_header_type(header_data.as_ptr() as *const c_void) };

        let variable_and_offset = {
            match header_type {
                OmHeaderType_t_OM_HEADER_LEGACY => Ok((header_data, None)),
                OmHeaderType_t_OM_HEADER_READ_TRAILER => unsafe {
                    let file_size = backend.count();
                    let trailer_size = om_trailer_size();
                    let trailer_offset = (file_size - trailer_size) as u64;
                    let owned_data = backend.get_bytes_owned(trailer_offset, trailer_size as u64);
                    let this_trailer = match owned_data {
                        Ok(ref data) => data.as_slice(),
                        Err(error) => backend.forward_unimplemented_error(error, || {
                            backend.get_bytes(trailer_offset, trailer_size as u64)
                        })?,
                    };
                    let mut offset = 0u64;
                    let mut size = 0u64;
                    if !om_trailer_read(
                        this_trailer.as_ptr() as *const c_void,
                        &mut offset,
                        &mut size,
                    ) {
                        return Err(OmFilesRsError::NotAnOmFile);
                    }

                    let offset_size = OmOffsetSize::new(offset, size);

                    let owned_data = backend.get_bytes_owned(offset, size);
                    let variable_data = match owned_data {
                        Ok(data) => data,
                        Err(error) => backend
                            .forward_unimplemented_error(error, || backend.get_bytes(offset, size))?
                            .to_vec(),
                    };
                    Ok((variable_data, Some(offset_size)))
                },
                OmHeaderType_t_OM_HEADER_INVALID => {
                    return Err(OmFilesRsError::NotAnOmFile);
                }
                _ => return Err(OmFilesRsError::NotAnOmFile),
            }
        };

        let (variable_data, offset_size) = variable_and_offset?;

        let variable_ptr = unsafe { om_variable_init(variable_data.as_ptr() as *const c_void) };
        Ok(Self {
            offset_size,
            backend,
            variable_data,
            variable: variable_ptr,
        })
    }

    pub fn data_type(&self) -> DataType {
        unsafe {
            DataType::try_from(om_variable_get_type(self.variable) as u8)
                .expect("Invalid data type")
        }
    }

    pub fn compression(&self) -> CompressionType {
        unsafe {
            CompressionType::try_from(om_variable_get_compression(self.variable) as u8)
                .expect("Invalid compression type")
        }
    }

    pub fn scale_factor(&self) -> f32 {
        unsafe { om_variable_get_scale_factor(self.variable) }
    }

    pub fn add_offset(&self) -> f32 {
        unsafe { om_variable_get_add_offset(self.variable) }
    }

    pub fn get_dimensions(&self) -> &[u64] {
        unsafe {
            let dims = om_variable_get_dimensions(self.variable);
            std::slice::from_raw_parts(dims.values, dims.count as usize)
        }
    }

    pub fn get_chunk_dimensions(&self) -> &[u64] {
        unsafe {
            let chunks = om_variable_get_chunks(self.variable);
            std::slice::from_raw_parts(chunks.values, chunks.count as usize)
        }
    }

    pub fn get_name(&self) -> Option<String> {
        unsafe {
            let name = om_variable_get_name(self.variable);
            if name.size == 0 {
                return None;
            }
            let bytes = std::slice::from_raw_parts(name.value as *const u8, name.size as usize);
            String::from_utf8(bytes.to_vec()).ok()
        }
    }

    /// Returns a HashMap mapping variable names to their offset and size
    /// This function needs to traverse the entire variable tree, therefore
    /// it is best to make sure that variable metadata is close to each other
    /// at the end of the file (before the trailer). The caller could then
    /// make sure that this part of the file is loaded/cached in memory
    pub fn get_flat_variable_metadata(&self) -> HashMap<String, (OmOffsetSize, bool)> {
        let mut result = HashMap::new();
        self.collect_variable_metadata(Vec::new(), &mut result);
        result
    }

    /// Helper function that recursively collects variable metadata
    fn collect_variable_metadata(
        &self,
        mut current_path: Vec<String>,
        result: &mut HashMap<String, (OmOffsetSize, bool)>,
    ) {
        // Add current variable's metadata if it has a name and offset_size
        // TODO: This requires for names to be unique
        if let Some(name) = self.get_name() {
            if let Some(offset_size) = &self.offset_size {
                let is_scalar = self.data_type().is_scalar();

                current_path.push(name.to_string());
                // Create hierarchical key
                let path_str = current_path
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("/");

                result.insert(path_str, (offset_size.clone(), is_scalar));
            }
        }

        // Process children
        let num_children = self.number_of_children();
        for i in 0..num_children {
            let child_path = current_path.clone();
            if let Some(child) = self.get_child(i) {
                child.collect_variable_metadata(child_path, result);
            }
        }
    }

    pub fn number_of_children(&self) -> u32 {
        unsafe { om_variable_get_children_count(self.variable) }
    }

    pub fn get_child(&self, index: u32) -> Option<Self> {
        let mut offset = 0u64;
        let mut size = 0u64;
        if !unsafe { om_variable_get_children(self.variable, index, 1, &mut offset, &mut size) } {
            return None;
        }

        let offset_size = OmOffsetSize::new(offset, size);
        let child = self
            .init_child_from_offset_size(offset_size)
            .expect("Failed to init child");
        Some(child)
    }

    pub fn init_child_from_offset_size(
        &self,
        offset_size: OmOffsetSize,
    ) -> Result<Self, OmFilesRsError> {
        let owned_data: Result<Vec<u8>, OmFilesRsError> = self
            .backend
            .get_bytes_owned(offset_size.offset, offset_size.size);
        let child_variable = match owned_data {
            Ok(data) => data,
            Err(error) => {
                let fallback_result = self.backend.forward_unimplemented_error(error, || {
                    self.backend.get_bytes(offset_size.offset, offset_size.size)
                })?;
                fallback_result.to_vec()
            }
        };

        let child_variable_ptr =
            unsafe { om_variable_init(child_variable.as_ptr() as *const c_void) };

        Ok(Self {
            offset_size: Some(offset_size),
            backend: self.backend.clone(),
            variable_data: child_variable,
            variable: child_variable_ptr,
        })
    }

    pub fn read_scalar<T: OmFileScalarDataType>(&self) -> Option<T> {
        if T::DATA_TYPE_SCALAR != self.data_type() {
            return None;
        }
        let mut value = T::default();

        let error =
            unsafe { om_variable_get_scalar(self.variable, &mut value as *mut T as *mut c_void) };

        if error != OmError_t_ERROR_OK {
            return None;
        }
        Some(value)
    }

    /// Read a variable as an array of a dynamic data type.
    pub fn read_into<T: OmFileArrayDataType>(
        &self,
        into: &mut ArrayD<T>,
        dim_read: &[Range<u64>],
        into_cube_offset: &[u64],
        into_cube_dimension: &[u64],
        io_size_max: Option<u64>,
        io_size_merge: Option<u64>,
    ) -> Result<(), OmFilesRsError> {
        let io_size_max = io_size_max.unwrap_or(65536);
        let io_size_merge = io_size_merge.unwrap_or(512);

        // Verify data type
        if T::DATA_TYPE_ARRAY != self.data_type() {
            return Err(OmFilesRsError::InvalidDataType);
        }

        let n_dimensions_read = dim_read.len();
        // TODO: Maybe cache this in the reader struct
        let n_dims = self.get_dimensions().len();

        // Validate dimension counts
        if n_dims != n_dimensions_read
            || n_dimensions_read != into_cube_offset.len()
            || n_dimensions_read != into_cube_dimension.len()
        {
            return Err(OmFilesRsError::MismatchingCubeDimensionLength);
        }

        // Prepare read parameters
        let read_offset: Vec<u64> = dim_read.iter().map(|r| r.start).collect();
        let read_count: Vec<u64> = dim_read.iter().map(|r| r.end - r.start).collect();

        // Initialize decoder
        let mut decoder = unsafe { create_uninit_decoder() };
        let error = unsafe {
            om_decoder_init(
                &mut decoder,
                self.variable,
                n_dimensions_read as u64,
                read_offset.as_ptr(),
                read_count.as_ptr(),
                into_cube_offset.as_ptr(),
                into_cube_dimension.as_ptr(),
                io_size_merge,
                io_size_max,
            )
        };

        if error != OmError_t_ERROR_OK {
            let error_string = c_error_string(error);
            return Err(OmFilesRsError::DecoderError(error_string));
        }

        // Allocate chunk buffer
        let chunk_buffer_size = unsafe { om_decoder_read_buffer_size(&decoder) };
        let mut chunk_buffer = Vec::<u8>::with_capacity(chunk_buffer_size as usize);

        // Perform decoding
        self.backend
            .decode(&mut decoder, into, chunk_buffer.as_mut_slice())?;

        Ok(())
    }

    pub fn read<T: OmFileArrayDataType + Clone + Zero>(
        &self,
        dim_read: &[Range<u64>],
        io_size_max: Option<u64>,
        io_size_merge: Option<u64>,
    ) -> Result<ArrayD<T>, OmFilesRsError> {
        let out_dims: Vec<u64> = dim_read.iter().map(|r| r.end - r.start).collect();
        let out_dims_usize = out_dims.iter().map(|&x| x as usize).collect::<Vec<_>>();

        let mut out = ArrayD::<T>::zeros(out_dims_usize);

        self.read_into::<T>(
            &mut out,
            dim_read,
            &vec![0; dim_read.len()],
            &out_dims,
            io_size_max,
            io_size_merge,
        )?;

        Ok(out)
    }
}

impl OmFileReader<MmapFile> {
    /// Convenience initializer to create an `OmFileReader` from a file path.
    pub fn from_file(file: &str) -> Result<Self, OmFilesRsError> {
        let file_handle = File::open(file).map_err(|e| OmFilesRsError::CannotOpenFile {
            filename: file.to_string(),
            errno: e.raw_os_error().unwrap_or(0),
            error: e.to_string(),
        })?;
        Self::from_file_handle(file_handle)
    }

    /// Convenience initializer to create an `OmFileReader` from an existing `FileHandle`.
    pub fn from_file_handle(file_handle: File) -> Result<Self, OmFilesRsError> {
        // TODO: Error handling
        let mmap = MmapFile::new(file_handle, Mode::ReadOnly).unwrap();
        Self::new(Arc::new(mmap)) // FIXME
    }

    /// Check if the file was deleted on the file system.
    /// Linux keeps the file alive as long as some processes have it open.
    pub fn was_deleted(&self) -> bool {
        self.backend.was_deleted()
    }
}
