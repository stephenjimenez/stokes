---
name: c-kernel-tooling
description: Procedures for compiling, debugging, and memory-checking C numerical kernels (stokes._core).
---

# C-Kernel Tooling & Memory Safety Runbook

Use this skill when modifying, debugging, or auditing C source files in `csrc/` (`assembly.c`, `kinematics.c`, `python_module.c`, and headers).

## 1. Standard In-Tree Compilation
After modifying any file in `csrc/` or `setup.py`, recompile the C extension in editable mode:

```bash
uv pip install -e . --no-build-isolation
```

Verify that the extension imports cleanly:
```bash
uv run python -c "import stokes._core; print('stokes._core loaded successfully')"
```

## 2. Debug & AddressSanitizer (ASan) Builds
To catch buffer overflows, memory corruptions, or illegal pointer offsets across the NumPy-C boundary:

```bash
CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
uv pip install -e . --no-build-isolation --force-reinstall
```

Then run the pytest suite or benchmark script with ASan symbols enabled:
```bash
uv run pytest -v
```

To switch back to optimized production builds:
```bash
uv pip install -e . --no-build-isolation --force-reinstall
```

## 3. Critical C-API Memory & Pointer Invariants
- **Multi-Dimensional Array Casts:**
  ```c
  const int (*cells)[8] = (const int (*)[8])PyArray_DATA(py_cells);
  const double (*x)[2] = (const double (*)[2])PyArray_DATA(py_x);
  const double (*dmg)[9] = (const double (*)[9])PyArray_DATA(py_dmg);
  ```
  Never introduce an extra asterisk (e.g. `const double*(*dmg)[9]` is invalid and rejected by modern C compilers).
- **Array Contiguity:**
  All NumPy arrays passed into `stokes._core` must be `C_CONTIGUOUS` (verified via `np.ascontiguousarray(...)`).
- **GIL Management:**
  C kernels must release the GIL for compute parallelism:
  ```c
  Py_BEGIN_ALLOW_THREADS
  ierr = assemble_stokes(...);
  Py_END_ALLOW_THREADS
  ```
  Never call Python C-API functions inside the `Py_BEGIN_ALLOW_THREADS` block.
