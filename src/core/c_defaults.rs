use om_file_format_sys::{
    om_decoder_init_data_read, om_decoder_init_index_read, om_error_string, OmDecoder_dataRead_t,
    OmDecoder_indexRead_t, OmDecoder_t, OmEncoder_t, OmError_t,
};

/// Create an uninitialized decoder.
/// You always need to call `om_decoder_init` before using the decoder!
pub unsafe fn create_uninit_decoder() -> OmDecoder_t {
    std::mem::zeroed()
}

/// Create an uninitialized encoder.
/// You always need to call `om_encoder_init` before using the encoder!
pub unsafe fn create_uninit_encoder() -> OmEncoder_t {
    std::mem::zeroed()
}

pub fn new_index_read(decoder: &OmDecoder_t) -> OmDecoder_indexRead_t {
    let mut index_read: OmDecoder_indexRead_t = unsafe { std::mem::zeroed() };
    unsafe { om_decoder_init_index_read(decoder, &mut index_read) };
    index_read
}

pub fn new_data_read(index_read: &OmDecoder_indexRead_t) -> OmDecoder_dataRead_t {
    let mut data_read: OmDecoder_dataRead_t = unsafe { std::mem::zeroed() };
    unsafe { om_decoder_init_data_read(&mut data_read, index_read) };
    data_read
}

pub fn c_error_string(error: OmError_t) -> String {
    let ptr = unsafe { om_error_string(error) };
    let error_string = unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() };
    error_string
}
