use crate::core::data_types::DataType;

pub fn to_numpy_dtype(dtype: &DataType) -> &str {
    match dtype {
        DataType::None => unimplemented!("todo scalar dtypes"),
        DataType::Int8 => unimplemented!("todo scalar dtypes"),
        DataType::Uint8 => unimplemented!("todo scalar dtypes"),
        DataType::Int16 => unimplemented!("todo scalar dtypes"),
        DataType::Uint16 => unimplemented!("todo scalar dtypes"),
        DataType::Int32 => unimplemented!("todo scalar dtypes"),
        DataType::Uint32 => unimplemented!("todo scalar dtypes"),
        DataType::Int64 => unimplemented!("todo scalar dtypes"),
        DataType::Uint64 => unimplemented!("todo scalar dtypes"),
        DataType::Float => unimplemented!("todo scalar dtypes"),
        DataType::Double => unimplemented!("todo scalar dtypes"),
        DataType::String => unimplemented!("todo scalar dtypes"),
        DataType::Int8Array => "int8",
        DataType::Uint8Array => "uint8",
        DataType::Int16Array => "int16",
        DataType::Uint16Array => "uint16",
        DataType::Int32Array => "int32",
        DataType::Uint32Array => "uint32",
        DataType::Int64Array => "int64",
        DataType::Uint64Array => "uint64",
        DataType::FloatArray => "float32",
        DataType::DoubleArray => "float64",
        DataType::StringArray => unimplemented!("todo string"),
    }
}
