use omfiles_rs::core::data_types::DataType;

pub fn to_numpy_dtype(dtype: &DataType) -> String {
    let scalar_or_array = ScalarOrArray::from_dtype(dtype);
    match scalar_or_array {
        ScalarOrArray::Scalar(s) => panic!("Scalar dtypes not supported: {}", s),
        ScalarOrArray::Array(s) => s,
    }
}

enum ScalarOrArray {
    Scalar(String),
    Array(String),
}

impl ScalarOrArray {
    fn from_dtype(dtype: &DataType) -> Self {
        match dtype {
            DataType::None => unimplemented!("todo None type"),
            DataType::Int8 => ScalarOrArray::Scalar("int8".to_string()),
            DataType::Uint8 => ScalarOrArray::Scalar("uint8".to_string()),
            DataType::Int16 => ScalarOrArray::Scalar("int16".to_string()),
            DataType::Uint16 => ScalarOrArray::Scalar("uint16".to_string()),
            DataType::Int32 => ScalarOrArray::Scalar("int32".to_string()),
            DataType::Uint32 => ScalarOrArray::Scalar("uint32".to_string()),
            DataType::Int64 => ScalarOrArray::Scalar("int64".to_string()),
            DataType::Uint64 => ScalarOrArray::Scalar("uint64".to_string()),
            DataType::Float => ScalarOrArray::Scalar("float32".to_string()),
            DataType::Double => ScalarOrArray::Scalar("float64".to_string()),
            DataType::String => ScalarOrArray::Scalar("string".to_string()),
            DataType::Int8Array => ScalarOrArray::Array("int8".to_string()),
            DataType::Uint8Array => ScalarOrArray::Array("uint8".to_string()),
            DataType::Int16Array => ScalarOrArray::Array("int16".to_string()),
            DataType::Uint16Array => ScalarOrArray::Array("uint16".to_string()),
            DataType::Int32Array => ScalarOrArray::Array("int32".to_string()),
            DataType::Uint32Array => ScalarOrArray::Array("uint32".to_string()),
            DataType::Int64Array => ScalarOrArray::Array("int64".to_string()),
            DataType::Uint64Array => ScalarOrArray::Array("uint64".to_string()),
            DataType::FloatArray => ScalarOrArray::Array("float32".to_string()),
            DataType::DoubleArray => ScalarOrArray::Array("float64".to_string()),
            DataType::StringArray => unimplemented!("todo string array type"),
        }
    }
}
