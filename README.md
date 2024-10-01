# Omfiles-rs Python bindings

## Development

```bash
# setup python virtual environment
python3 -m venv env
source env/bin/activate

pip install maturin
maturin develop
# if you encounter an error:  Both VIRTUAL_ENV and CONDA_PREFIX are set. Please unset one of them
unset CONDA_PREFIX
```

### Tests

```bash
cargo test
```

## Usage

```python
import omfilesrspy

omfilesrspy.read_om_file("test_file.om")
```
