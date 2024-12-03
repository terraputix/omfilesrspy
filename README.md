# Omfiles-rs Python bindings

## Development

```bash
# setup python virtual environment
python3 -m venv env
source env/bin/activate

pip install maturin numpy
maturin develop
# if you encounter an error:  Both VIRTUAL_ENV and CONDA_PREFIX are set. Please unset one of them
unset CONDA_PREFIX
```

### Tests

```bash
cargo test --no-default-features
```

## Usage

```python
import omfilesrspy

omfilesrspy.read_om_file("test_file.om")
```

## Benchmarks

Before running the benchmarks, make sure to compile the release version of the library:

```bash
maturin develop --release
```

Then run the benchmarks:

```bash
python benchmarks/benchmarks.py
```
