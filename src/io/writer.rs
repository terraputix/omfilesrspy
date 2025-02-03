use crate::backend::backends::OmFileWriterBackend;
use crate::core::c_defaults::{c_error_string, create_uninit_encoder};
use crate::core::compression::CompressionType;
use crate::core::data_types::{DataType, OmFileArrayDataType, OmFileScalarDataType};
use crate::errors::OmFilesRsError;
use crate::io::buffered_writer::OmBufferedWriter;
use ndarray::ArrayViewD;
use om_file_format_sys::{
    om_encoder_chunk_buffer_size, om_encoder_compress_chunk, om_encoder_compress_lut,
    om_encoder_compressed_chunk_buffer_size, om_encoder_count_chunks,
    om_encoder_count_chunks_in_array, om_encoder_init, om_encoder_lut_buffer_size, om_header_write,
    om_header_write_size, om_trailer_size, om_trailer_write, om_variable_write_numeric_array,
    om_variable_write_numeric_array_size, om_variable_write_scalar, om_variable_write_scalar_size,
    OmEncoder_t, OmError_t_ERROR_OK,
};
use std::borrow::BorrowMut;
use std::marker::PhantomData;
use std::os::raw::c_void;

#[derive(Debug, Clone, PartialEq)]
pub struct OmOffsetSize {
    pub offset: u64,
    pub size: u64,
}

impl OmOffsetSize {
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }
}

pub struct OmFileWriter<Backend: OmFileWriterBackend> {
    buffer: OmBufferedWriter<Backend>,
}

impl<Backend: OmFileWriterBackend> OmFileWriter<Backend> {
    pub fn new(backend: Backend, initial_capacity: u64) -> Self {
        Self {
            buffer: OmBufferedWriter::new(backend, initial_capacity as usize),
        }
    }

    pub fn write_header_if_required(&mut self) -> Result<(), OmFilesRsError> {
        if self.buffer.total_bytes_written > 0 {
            return Ok(());
        }
        let size = unsafe { om_header_write_size() };
        self.buffer.reallocate(size as usize)?;
        unsafe {
            om_header_write(self.buffer.buffer_at_write_position().as_mut_ptr() as *mut c_void);
        }
        self.buffer.increment_write_position(size as usize);
        Ok(())
    }

    pub fn write_scalar<T: OmFileScalarDataType>(
        &mut self,
        value: T,
        name: &str,
        children: &[OmOffsetSize],
    ) -> Result<OmOffsetSize, OmFilesRsError> {
        self.write_header_if_required()?;

        assert!(name.len() <= u16::MAX as usize);
        assert!(children.len() <= u32::MAX as usize);

        let type_scalar = T::DATA_TYPE_SCALAR.to_c();

        let size = unsafe {
            om_variable_write_scalar_size(name.len() as u16, children.len() as u32, type_scalar)
        };

        self.buffer.align_to_64_bytes()?;
        let offset = self.buffer.total_bytes_written as u64;

        self.buffer.reallocate(size)?;

        let children_offsets: Vec<u64> = children.iter().map(|c| c.offset).collect();
        let children_sizes: Vec<u64> = children.iter().map(|c| c.size).collect();
        unsafe {
            om_variable_write_scalar(
                self.buffer.buffer_at_write_position().as_mut_ptr() as *mut c_void,
                name.len() as u16,
                children.len() as u32,
                children_offsets.as_ptr(),
                children_sizes.as_ptr(),
                name.as_ptr() as *const ::std::os::raw::c_char,
                type_scalar,
                &value as *const T as *const c_void,
            )
        };

        self.buffer.increment_write_position(size);
        Ok(OmOffsetSize::new(offset, size as u64))
    }

    pub fn prepare_array<T: OmFileArrayDataType>(
        &mut self,
        dimensions: Vec<u64>,
        chunk_dimensions: Vec<u64>,
        compression: CompressionType,
        scale_factor: f32,
        add_offset: f32,
    ) -> Result<OmFileWriterArray<T, Backend>, OmFilesRsError> {
        let _ = &self.write_header_if_required()?;

        let array_writer = OmFileWriterArray::new(
            dimensions,
            chunk_dimensions,
            compression,
            T::DATA_TYPE_ARRAY,
            scale_factor,
            add_offset,
            self.buffer.borrow_mut(),
        )?;

        Ok(array_writer)
    }

    pub fn write_array(
        &mut self,
        array: &OmFileWriterArrayFinalized,
        name: &str,
        children: &[OmOffsetSize],
    ) -> Result<OmOffsetSize, OmFilesRsError> {
        self.write_header_if_required()?;

        debug_assert!(name.len() <= u16::MAX as usize);
        debug_assert_eq!(array.dimensions.len(), array.chunks.len());

        let size = unsafe {
            om_variable_write_numeric_array_size(
                name.len() as u16,
                children.len() as u32,
                array.dimensions.len() as u64,
            )
        };
        self.buffer.align_to_64_bytes()?;

        let offset = self.buffer.total_bytes_written as u64;

        self.buffer.reallocate(size)?;

        let children_offsets: Vec<u64> = children.iter().map(|c| c.offset).collect();
        let children_sizes: Vec<u64> = children.iter().map(|c| c.size).collect();
        unsafe {
            om_variable_write_numeric_array(
                self.buffer.buffer_at_write_position().as_mut_ptr() as *mut c_void,
                name.len() as u16,
                children.len() as u32,
                children_offsets.as_ptr(),
                children_sizes.as_ptr(),
                name.as_ptr() as *const ::std::os::raw::c_char,
                array.data_type.to_c(),
                array.compression.to_c(),
                array.scale_factor,
                array.add_offset,
                array.dimensions.len() as u64,
                array.dimensions.as_ptr(),
                array.chunks.as_ptr(),
                array.lut_size,
                array.lut_offset,
            )
        };

        self.buffer.increment_write_position(size);
        Ok(OmOffsetSize::new(offset, size as u64))
    }

    pub fn write_trailer(&mut self, root_variable: OmOffsetSize) -> Result<(), OmFilesRsError> {
        self.write_header_if_required()?;
        self.buffer.align_to_64_bytes()?;

        let size = unsafe { om_trailer_size() };
        self.buffer.reallocate(size)?;
        unsafe {
            om_trailer_write(
                self.buffer.buffer_at_write_position().as_mut_ptr() as *mut c_void,
                root_variable.offset,
                root_variable.size,
            );
        }
        self.buffer.increment_write_position(size);

        self.buffer.write_to_file()
    }
}

pub struct OmFileWriterArray<'a, OmType: OmFileArrayDataType, Backend: OmFileWriterBackend> {
    look_up_table: Vec<u64>,
    encoder: OmEncoder_t,
    chunk_index: u64,
    scale_factor: f32,
    add_offset: f32,
    compression: CompressionType,
    data_type: PhantomData<OmType>,
    dimensions: Vec<u64>,
    chunks: Vec<u64>,
    compressed_chunk_buffer_size: u64,
    chunk_buffer: Vec<u8>,
    buffer: &'a mut OmBufferedWriter<Backend>,
}

impl<'a, OmType: OmFileArrayDataType, Backend: OmFileWriterBackend>
    OmFileWriterArray<'a, OmType, Backend>
{
    /// `lut_chunk_element_count` should be 256 for production files.
    pub fn new(
        dimensions: Vec<u64>,
        chunk_dimensions: Vec<u64>,
        compression: CompressionType,
        data_type: DataType,
        scale_factor: f32,
        add_offset: f32,
        buffer: &'a mut OmBufferedWriter<Backend>,
    ) -> Result<Self, OmFilesRsError> {
        if data_type != OmType::DATA_TYPE_ARRAY {
            return Err(OmFilesRsError::InvalidDataType);
        }
        if dimensions.len() != chunk_dimensions.len() {
            return Err(OmFilesRsError::MismatchingCubeDimensionLength);
        }

        let chunks = chunk_dimensions;

        let mut encoder = unsafe { create_uninit_encoder() };
        let error = unsafe {
            om_encoder_init(
                &mut encoder,
                scale_factor,
                add_offset,
                compression.to_c(),
                data_type.to_c(),
                dimensions.as_ptr(),
                chunks.as_ptr(),
                dimensions.len() as u64,
            )
        };
        if error != OmError_t_ERROR_OK {
            return Err(OmFilesRsError::FileWriterError {
                errno: error as i32,
                error: c_error_string(error),
            });
        }

        let n_chunks = unsafe { om_encoder_count_chunks(&encoder) } as usize;
        let compressed_chunk_buffer_size =
            unsafe { om_encoder_compressed_chunk_buffer_size(&encoder) };
        let chunk_buffer_size = unsafe { om_encoder_chunk_buffer_size(&encoder) } as usize;

        let chunk_buffer = vec![0u8; chunk_buffer_size];
        let look_up_table = vec![0u64; n_chunks + 1];

        Ok(Self {
            look_up_table,
            encoder,
            chunk_index: 0,
            scale_factor,
            add_offset,
            compression,
            data_type: PhantomData,
            dimensions,
            chunks,
            compressed_chunk_buffer_size,
            chunk_buffer,
            buffer,
        })
    }

    /// Writes an ndarray to the file.
    pub fn write_data(
        &mut self,
        array: ArrayViewD<OmType>,
        array_offset: Option<&[u64]>,
        array_count: Option<&[u64]>,
    ) -> Result<(), OmFilesRsError> {
        let array_dimensions = array
            .shape()
            .iter()
            .map(|&x| x as u64)
            .collect::<Vec<u64>>();
        let array = array.as_slice().ok_or(OmFilesRsError::ArrayNotContiguous)?;
        self.write_data_flat(array, Some(&array_dimensions), array_offset, array_count)
    }

    /// Compresses data and writes it to file.
    pub fn write_data_flat(
        &mut self,
        array: &[OmType],
        array_dimensions: Option<&[u64]>,
        array_offset: Option<&[u64]>,
        array_count: Option<&[u64]>,
    ) -> Result<(), OmFilesRsError> {
        let array_dimensions = array_dimensions.unwrap_or(&self.dimensions);
        let default_offset = vec![0; array_dimensions.len()];
        let array_offset = array_offset.unwrap_or(default_offset.as_slice());
        let array_count = array_count.unwrap_or(array_dimensions);

        if array_count.len() != self.dimensions.len() {
            return Err(OmFilesRsError::ChunkHasWrongNumberOfElements);
        }
        for (array_dim, max_dim) in array_count.iter().zip(self.dimensions.iter()) {
            if array_dim > max_dim {
                return Err(OmFilesRsError::ChunkHasWrongNumberOfElements);
            }
        }

        let array_size: u64 = array_dimensions.iter().product::<u64>();
        if array.len() as u64 != array_size {
            return Err(OmFilesRsError::ChunkHasWrongNumberOfElements);
        }
        for (dim, (offset, count)) in array_dimensions
            .iter()
            .zip(array_offset.iter().zip(array_count.iter()))
        {
            if offset + count > *dim {
                return Err(OmFilesRsError::OffsetAndCountExceedDimension {
                    offset: *offset,
                    count: *count,
                    dimension: *dim,
                });
            }
        }

        self.buffer
            .reallocate(self.compressed_chunk_buffer_size as usize * 4)?;

        let number_of_chunks_in_array =
            unsafe { om_encoder_count_chunks_in_array(&mut self.encoder, array_count.as_ptr()) };

        if self.chunk_index == 0 {
            self.look_up_table[self.chunk_index as usize] = self.buffer.total_bytes_written as u64;
        }

        // This loop could be parallelized. However, the order of chunks must
        // remain the same in the LUT and final output buffer.
        // For multithreading, we would need multiple buffers that need to be
        // copied into the final buffer in the correct order after compression.
        for chunk_offset in 0..number_of_chunks_in_array {
            self.buffer
                .reallocate(self.compressed_chunk_buffer_size as usize)?;

            let bytes_written = unsafe {
                om_encoder_compress_chunk(
                    &mut self.encoder,
                    array.as_ptr() as *const c_void,
                    array_dimensions.as_ptr(),
                    array_offset.as_ptr(),
                    array_count.as_ptr(),
                    self.chunk_index,
                    chunk_offset,
                    self.buffer.buffer_at_write_position().as_mut_ptr(),
                    self.chunk_buffer.as_mut_ptr(),
                )
            };

            self.buffer.increment_write_position(bytes_written as usize);

            self.look_up_table[(self.chunk_index + 1) as usize] =
                self.buffer.total_bytes_written as u64;
            self.chunk_index += 1;
        }

        Ok(())
    }

    /// Compress the lookup table and write it to the output buffer.
    pub fn write_lut(&mut self) -> u64 {
        let buffer_size = unsafe {
            om_encoder_lut_buffer_size(self.look_up_table.as_ptr(), self.look_up_table.len() as u64)
        };

        self.buffer
            .reallocate(buffer_size as usize)
            .expect("Failed to reallocate buffer");

        let compressed_lut_size = unsafe {
            om_encoder_compress_lut(
                self.look_up_table.as_ptr(),
                self.look_up_table.len() as u64,
                self.buffer.buffer_at_write_position().as_mut_ptr(),
                buffer_size,
            )
        };

        self.buffer
            .increment_write_position(compressed_lut_size as usize);
        compressed_lut_size
    }

    /// Finalize the array and return the finalized struct.
    pub fn finalize(mut self) -> OmFileWriterArrayFinalized {
        let lut_offset = self.buffer.total_bytes_written as u64;
        let lut_size = self.write_lut();

        OmFileWriterArrayFinalized {
            scale_factor: self.scale_factor,
            add_offset: self.add_offset,
            compression: self.compression,
            data_type: OmType::DATA_TYPE_ARRAY,
            dimensions: self.dimensions.clone(),
            chunks: self.chunks.clone(),
            lut_size,
            lut_offset,
        }
    }
}

pub struct OmFileWriterArrayFinalized {
    pub scale_factor: f32,
    pub add_offset: f32,
    pub compression: CompressionType,
    pub data_type: DataType,
    pub dimensions: Vec<u64>,
    pub chunks: Vec<u64>,
    pub lut_size: u64,
    pub lut_offset: u64,
}
