#[cfg(test)]
pub use utils::*;

#[cfg(test)]
mod utils {
    use std::fs;
    use std::path::Path;

    // Helper function to ensure test directory exists
    pub fn ensure_test_dir() -> std::io::Result<()> {
        fs::create_dir_all("test_files")?;
        Ok(())
    }

    // Generate a simple binary file with specified bytes
    pub fn create_binary_file(filename: &str, data: &[u8]) -> std::io::Result<()> {
        ensure_test_dir()?;
        let path = Path::new("test_files").join(filename);
        fs::write(path, data)?;
        Ok(())
    }
}
