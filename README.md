# Python bindings for Open Meteo file format

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/omfiles.svg)](https://pypi.org/project/omfiles/)
[![Build and Test](https://github.com/terraputix/omfilesrspy/actions/workflows/build-test.yml/badge.svg)](https://github.com/terraputix/omfilesrspy/actions/workflows/build-test.yml)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Publish](https://github.com/terraputix/omfilesrspy/actions/workflows/publish.yml/badge.svg)](https://github.com/terraputix/omfilesrspy/actions/workflows/publish.yml)

## Installation

> **Note:** This package is currently under active development and not yet ready for production use. APIs may change without notice until the first stable release.

```bash
pip install omfiles
```

## Features

- Fast reading and writing of multi-dimensional arrays
- Hierarchical data structure support
- Integration with [NumPy](https://github.com/numpy/numpy) arrays
- Chunked data access for efficient I/O
- Support for [fsspec](https://github.com/fsspec/filesystem_spec) and [xarray](https://github.com/pydata/xarray)

### Basic Reading

OM files are [structured like a tree of variables](https://github.com/open-meteo/om-file-format?tab=readme-ov-file#data-hierarchy-model). The following example assumes that the file `test_file.om` contains an array variable as a root variable which has a dimensionality greater than 2 and a size of at least 2x100:

```python
from omfiles import OmFilePyReader

reader = OmFilePyReader("test_file.om")
data = reader[0:2, 0:100, ...]
```

### Writing Arrays

#### Simple Array
```python
import numpy as np
from omfiles import OmFilePyWriter

# Create sample data
data = np.random.rand(100, 100).astype(np.float32)

# Initialize writer
writer = OmFilePyWriter("simple.om")

# Write array with compression
variable = writer.write_array(
    data,
    chunks=[50, 50],
    scale_factor=1.0,
    add_offset=0.0,
    compression="pfor_delta_2d",
    name="data"
)

# Finalize the file. This writes the trailer and flushes the buffers.
writer.close(variable)
```

#### Hierarchical Structure
```python
import numpy as np
from omfiles import OmFilePyWriter

# Create sample data
features = np.random.rand(1000, 64).astype(np.float32)
labels = np.random.randint(0, 10, size=(1000,), dtype=np.int32)

# Initialize writer
writer = OmFilePyWriter("hierarchical.om")

# Write child arrays first
features_var = writer.write_array(
    features,
    chunks=[100, 64],
    name="features",
    compression="pfor_delta_2d"
)

labels_var = writer.write_array(
    labels,
    chunks=[100],
    name="labels"
)

metadata_var = writer.write_scalar(
    42,
    name="metadata"
)

# Create root group with children
root_var = writer.write_scalar(
    0, # This is just placeholder data, later we will support creating groups with no data
    name="root",
    children=[features_var, labels_var, metadata_var]
)

# Finalize the file
writer.close(root_var)
```


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

## Benchmarks

Before running the benchmarks, make sure to compile the release version of the library:

```bash
maturin develop --release
```

Then run the benchmarks:

```bash
python benchmarks/main.py
```
