# pyhccl

Python bindings for the Ascend **[HCCL](https://gitcode.com/cann/hccl)** /
**[HCOMM](https://gitcode.com/cann/hcomm)** collective-communication libraries.

`pyhccl` exposes the full set of host-side C interfaces from the `include`
directories of the `hccl` and `hcomm` repositories as Python functions, and
mirrors their enums and structs. It is built with **pybind11** and follows the
same pattern as PyTorch's `torch._C` extension: the native libraries are loaded
lazily via `dlopen`, so the package can be imported and inspected even on a
machine without the CANN runtime.

## Features

- Covers **all** host-side C APIs under `hccl/include` and `hcomm/include`:
  collectives, `SEND/RECV`, communicator management, channels, threads,
  symmetric windows, rank-graph/topology queries, teams, CCU resources, and
  the data-plane primitives.
- Exposes every enum, struct, and handle type as a Python counterpart.
- Lazy native-library loading (`dlopen` / `dlsym`); import never fails.
- Pure ABI bindings — no source dependency on the ACL/SecureC private headers,
  so it builds on a clean machine (e.g. CI / manylinux).

## Requirements

- A C++17 compiler (the wheel build only needs the compiler, **not** the CANN SDK).
- The native `libhccl.so` / `libhcomm.so` at runtime, only if you actually call a
  communication function.

## Install

```bash
pip install pyhccl
```

For a local editable install:

```bash
pip install -e .
```

## Quick start

```python
import pyhccl

# Optional: load the native library explicitly (path via PYHCCL_LIBRARY or
# PYHCCL_LIBRARY_PATH). Functions load it automatically on first use.
n_loaded = pyhccl.load_library()

# Enums are available both as module-level constants and as enum members.
assert pyhccl.HCCL_SUCCESS == 0
assert int(pyhccl.HcclDataType.HCCL_DATA_TYPE_FP32) == 4

# Structs are first-class Python objects.
config = pyhccl.HcclCommConfig()
pyhccl.HcclCommConfigInit(config)
config.hcclCommName = "my_comm"

# Functions with output parameters return (ret, out0, out1, ...).
ret, rank_size = pyhccl.HcclGetRankSize(comm)
if ret != pyhccl.HCCL_SUCCESS:
    raise RuntimeError(pyhccl.get_error_string(ret))
```

## Design

```
src/pyhccl/__init__.py      # Python package, re-exports pyhccl._C
src/csrc/abi.hpp            # ABI mirror of the hccl/hcomm headers (enums/structs)
src/csrc/wrappers.hpp       # dlsym-based forwarders for every C function
src/csrc/loader.{hpp,cpp}   # dlopen / dlsym lazy loader
src/csrc/bindings.cpp       # pybind11 module (pyhccl._C)
```

- All raw pointers (handles, buffers, streams) are represented in Python as
  `int` addresses; on x86-64/aarch64 this is ABI-identical to `void *`.
- Code comments are written in Chinese; runtime log/error messages in English.

### Scope note

The bindings cover every host-side C interface (`extern` functions and
`static inline` initializers) in the `include` directories. The `ccu/*.hpp`
files are a C++ device-side kernel DSL (templates that expand against the
private `ccu_primitives_impl.h`); they are compiled into kernels rather than
called at runtime, so they are intentionally not exposed as Python functions.
Their host-side CCU resource/launch APIs (`ccu_res.h`, `ccu_launch.h`,
`ccu_types.h` enums) **are** bound.

## Building & testing

```bash
pip install -e .           # build the C++ extension
pip install pytest
pytest tests
```

Building the wheel:

```bash
pip install build
python -m build
```

## Linting

```bash
clang-format --dry-run --Werror src/csrc/*.hpp src/csrc/*.cpp  # C++ (see .clang-format)
ruff check src tests && ruff format --check src tests           # Python (see [tool.ruff])
pylint src/pyhccl tests                                         # Python (see .pylintrc)
```

These checks also run in the [lint workflow](.github/workflows/lint.yml), which pins
exact tool versions (clang-format 21.1.0, ruff 0.17.1, pylint 4.1.3). Use matching
versions locally to avoid false positives (older clang-format, e.g. 18.x from apt,
formats a few lines differently).

## License

Apache License 2.0. See [LICENSE](LICENSE).
