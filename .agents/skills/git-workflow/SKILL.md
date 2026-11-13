---
name: git-workflow
description: Standardized Git workflow, subagent branch isolation, and conventional commit guidelines.
---

# Git Workflow & Subagent Branch Management

Use this skill when managing git branches, handling subagent workspace merges, or committing changes in the `stokes` repository.

## 1. Subagent Workspace Isolation
Antigravity subagents implementing substantial changes should be launched with `Workspace: "branch"`.
- This ensures code modifications remain in an isolated git branch and do not pollute `main` until audited and approved.
- The Critic reviews the diff against `main`.

## 2. Conventional Commit Guidelines
All commits should follow the Conventional Commits specification:

| Type | When to Use | Example |
| :--- | :--- | :--- |
| `feat` | New solver capabilities, elements, or damage formulations | `feat(fe): add Robin boundary condition support` |
| `fix` | Bug fixes in C kernels, assembly, or Python driver | `fix(csrc): correct dmg pointer cast in python_module` |
| `perf` | Computational speedups in C quadrature or assembly | `perf(csrc): vectorize Q8 serendipity shape derivatives` |
| `test` | Adding or updating unit tests and verification benchmarks | `test(shape): add Kronecker delta tests for Q8 nodes` |
| `docs` | Documentation, docstrings, or TOML parameter guides | `docs(readme): document Taylor-Hood DOF ordering` |
| `refactor`| Code restructuring without behavioral changes | `refactor(assembly): streamline CSC matrix bincount map` |

## 3. Pre-Merge Verification Checklist
Before merging any feature branch into `main`:
1. [ ] Recompile C extension: `uv pip install -e . --no-build-isolation`
2. [ ] Passing unit tests: `uv run pytest -v`
3. [ ] Passing analytical benchmark: `uv run python .agents/skills/fem-verifier/scripts/verify_gravity_solve.py`
4. [ ] Clean code formatting: `uv run ruff check && uv run ruff format --check`
