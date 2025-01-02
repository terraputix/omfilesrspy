# Omfiles-rs Python bindings

## Development

```bash
# setup python virtual environment with pyenv
python -m venv .venv
source .venv/bin/activate
# To always activate this environment in this directory run `pyenv local pyo3`
pip install maturin

maturin develop --extras=dev
# if you encounter an error:  Both VIRTUAL_ENV and CONDA_PREFIX are set. Please unset one of them
unset CONDA_PREFIX
```

### Tests

```bash
cargo test --no-default-features
```

## Usage

```python
from omfilesrspy import OmFilePyReader

reader = OmFilePyReader("test_file.om")
data = reader[0:2, 0:100, ...]
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

## Benchmarks

Before running the benchmarks, make sure to compile the release version of the library:

```bash
maturin develop --release
```

Then run the benchmarks:

```bash
python benchmarks/benchmarks.py
```
