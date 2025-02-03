use crate::backend::mmapfile::{MAdvice, MmapFile, MmapType};
use crate::core::c_defaults::{c_error_string, new_data_read, new_index_read};
use crate::core::data_types::OmFileArrayDataType;
use crate::errors::OmFilesRsError;
use ndarray::ArrayD;
use om_file_format_sys::{
    om_decoder_decode_chunks, om_decoder_next_data_read, om_decoder_next_index_read, OmDecoder_t,
    OmError_t_ERROR_OK,
};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::os::raw::c_void;

pub trait OmFileWriterBackend {
    fn write(&mut self, data: &[u8]) -> Result<(), OmFilesRsError>;
    fn write_at(&mut self, data: &[u8], offset: usize) -> Result<(), OmFilesRsError>;
    fn synchronize(&self) -> Result<(), OmFilesRsError>;
}

/// A trait for reading byte data from different storage backends.
/// Provides methods for reading bytes either by reference or as owned data,
/// as well as functions for prefetching and pre-reading data.
pub trait OmFileReaderBackend {
    /// Length in bytes
    fn count(&self) -> usize;
    fn needs_prefetch(&self) -> bool;
    fn prefetch_data(&self, offset: usize, count: usize);
    fn pre_read(&self, offset: usize, count: usize) -> Result<(), OmFilesRsError>;

    /// Returns a reference to a slice of bytes from the backend, starting at `offset` and reading `count` bytes.
    /// At least one of `get_bytes` or `get_bytes_owned` must be implemented.
    fn get_bytes(&self, _offset: u64, _count: u64) -> Result<&[u8], OmFilesRsError> {
        Err(OmFilesRsError::NotImplementedError(
            "You need to implement either get_bytes or get_bytes_owned!".to_string(),
        ))
    }

    /// Returns an owned Vec<u8> containing bytes from the backend, starting at `offset` and reading `count` bytes.
    /// At least one of `get_bytes` or `get_bytes_owned` must be implemented.
    fn get_bytes_owned(&self, _offset: u64, _count: u64) -> Result<Vec<u8>, OmFilesRsError> {
        Err(OmFilesRsError::NotImplementedError(
            "You need to implement either get_bytes or get_bytes_owned!".to_string(),
        ))
    }

    fn forward_unimplemented_error<'a, F>(
        &'a self,
        e: OmFilesRsError,
        fallback_fn: F,
    ) -> Result<&'a [u8], OmFilesRsError>
    where
        F: FnOnce() -> Result<&'a [u8], OmFilesRsError>,
    {
        match e {
            OmFilesRsError::NotImplementedError(_) => fallback_fn(),
            _ => Err(e),
        }
    }

    fn decode<OmType: OmFileArrayDataType>(
        &self,
        decoder: &OmDecoder_t,
        into: &mut ArrayD<OmType>,
        chunk_buffer: &mut [u8],
    ) -> Result<(), OmFilesRsError> {
        #[allow(unused_mut)]
        let mut into = into
            .as_slice_mut()
            .ok_or(OmFilesRsError::ArrayNotContiguous)?;

        let mut index_read = new_index_read(decoder);
        unsafe {
            // Loop over index blocks and read index data
            while om_decoder_next_index_read(decoder, &mut index_read) {
                // Get bytes for index-read as owned data or as reference
                let owned_data = self.get_bytes_owned(index_read.offset, index_read.count);
                let index_data = match owned_data {
                    Ok(ref data) => data.as_slice(),
                    Err(error) => self.forward_unimplemented_error(error, || {
                        self.get_bytes(index_read.offset, index_read.count)
                    })?,
                };

                let mut data_read = new_data_read(&index_read);

                let mut error = OmError_t_ERROR_OK;

                // Loop over data blocks and read compressed data chunks
                while om_decoder_next_data_read(
                    decoder,
                    &mut data_read,
                    index_data.as_ptr() as *const c_void,
                    index_read.count,
                    &mut error,
                ) {
                    // Get bytes for data-read as owned data or as reference
                    let owned_data = self.get_bytes_owned(data_read.offset, data_read.count);
                    let data_data = match owned_data {
                        Ok(ref data) => data.as_slice(),
                        Err(error) => self.forward_unimplemented_error(error, || {
                            self.get_bytes(data_read.offset, data_read.count)
                        })?,
                    };

                    if !om_decoder_decode_chunks(
                        decoder,
                        data_read.chunkIndex,
                        data_data.as_ptr() as *const c_void,
                        data_read.count,
                        into.as_mut_ptr() as *mut c_void,
                        chunk_buffer.as_mut_ptr() as *mut c_void,
                        &mut error,
                    ) {
                        let error_string = c_error_string(error);
                        return Err(OmFilesRsError::DecoderError(error_string));
                    }
                }
                if error != OmError_t_ERROR_OK {
                    let error_string = c_error_string(error);
                    return Err(OmFilesRsError::DecoderError(error_string));
                }
            }
        }
        Ok(())
    }
}

fn map_io_error(e: std::io::Error) -> OmFilesRsError {
    OmFilesRsError::FileWriterError {
        errno: e.raw_os_error().unwrap_or(0),
        error: e.to_string(),
    }
}

impl OmFileWriterBackend for &File {
    fn write(&mut self, data: &[u8]) -> Result<(), OmFilesRsError> {
        self.write_all(data).map_err(|e| map_io_error(e))?;
        Ok(())
    }

    fn write_at(&mut self, data: &[u8], offset: usize) -> Result<(), OmFilesRsError> {
        self.seek(SeekFrom::Start(offset as u64))
            .map_err(|e| map_io_error(e))?;
        self.write_all(data).map_err(|e| map_io_error(e))?;
        Ok(())
    }

    fn synchronize(&self) -> Result<(), OmFilesRsError> {
        self.sync_all().map_err(|e| map_io_error(e))?;
        Ok(())
    }
}

impl OmFileWriterBackend for File {
    fn write(&mut self, data: &[u8]) -> Result<(), OmFilesRsError> {
        self.write_all(data).map_err(|e| map_io_error(e))?;
        Ok(())
    }

    fn write_at(&mut self, data: &[u8], offset: usize) -> Result<(), OmFilesRsError> {
        self.seek(SeekFrom::Start(offset as u64))
            .map_err(|e| map_io_error(e))?;
        self.write_all(data).map_err(|e| map_io_error(e))?;
        Ok(())
    }

    fn synchronize(&self) -> Result<(), OmFilesRsError> {
        self.sync_all().map_err(|e| map_io_error(e))?;
        Ok(())
    }
}

impl OmFileReaderBackend for MmapFile {
    fn count(&self) -> usize {
        self.data.len()
    }

    fn needs_prefetch(&self) -> bool {
        true
    }

    fn prefetch_data(&self, offset: usize, count: usize) {
        self.prefetch_data_advice(offset, count, MAdvice::WillNeed);
    }

    fn pre_read(&self, _offset: usize, _count: usize) -> Result<(), OmFilesRsError> {
        // No-op for mmaped file
        Ok(())
    }

    fn get_bytes(&self, offset: u64, count: u64) -> Result<&[u8], OmFilesRsError> {
        let index_range = (offset as usize)..(offset + count) as usize;
        match self.data {
            MmapType::ReadOnly(ref mmap) => Ok(&mmap[index_range]),
            MmapType::ReadWrite(ref mmap_mut) => Ok(&mmap_mut[index_range]),
        }
    }
}

#[derive(Debug)]
pub struct InMemoryBackend {
    data: Vec<u8>,
}

impl InMemoryBackend {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }
}

impl OmFileWriterBackend for &mut InMemoryBackend {
    fn write(&mut self, data: &[u8]) -> Result<(), OmFilesRsError> {
        self.data.extend_from_slice(data);
        Ok(())
    }

    fn write_at(&mut self, data: &[u8], offset: usize) -> Result<(), OmFilesRsError> {
        self.data.reserve(offset + data.len());
        let dst = &mut self.data[offset..offset + data.len()];
        dst.copy_from_slice(data);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), OmFilesRsError> {
        // No-op for in-memory backend
        Ok(())
    }
}

impl OmFileReaderBackend for InMemoryBackend {
    fn count(&self) -> usize {
        self.data.len()
    }

    fn needs_prefetch(&self) -> bool {
        false
    }

    fn prefetch_data(&self, _offset: usize, _count: usize) {
        // No-op for in-memory backend
    }

    fn pre_read(&self, _offset: usize, _count: usize) -> Result<(), OmFilesRsError> {
        // No-op for in-memory backend
        Ok(())
    }

    fn get_bytes(&self, offset: u64, count: u64) -> Result<&[u8], OmFilesRsError> {
        let index_range = (offset as usize)..(offset + count) as usize;
        Ok(&self.data[index_range])
    }
}
