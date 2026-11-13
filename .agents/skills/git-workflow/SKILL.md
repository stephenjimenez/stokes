---
name: git-workflow
description: Standardized Git and GitHub CLI (gh) workflow, issue branch naming, subagent workspace rules, conventional commits, and PR creation.
---

# Git & GitHub Workflow

Use this skill when reading GitHub issues, creating branches, managing subagent workspaces, committing changes, or pushing PRs in the `stokes` repository.

## 1. Issue Intake & Branch Creation
When starting work from a GitHub issue:
1. Read the full issue description and comments:
   ```bash
   gh issue view <ISSUE_NUMBER> --comments
   ```
2. Sync `main` and create an issue-linked branch using a Conventional Commit prefix (`feat`, `fix`, `perf`, `refactor`, `test`, `docs`):
   ```bash
   git checkout main && git pull --ff-only
   git checkout -b <type>/issue-<ISSUE_NUMBER>-<short-slug>
   ```

## 2. Subagent Workspace Strategy
Avoid double-branching when orchestrating subagents:
- **Issue-Driven Workflow (Recommended):** The primary agent checks out `<type>/issue-<ISSUE_NUMBER>-<short-slug>` first, and `generator` runs with `workspace: inherit` so all edits land directly on the issue branch.
- **Ad-Hoc Task on `main`:** If starting directly on `main` without an issue branch, launch `generator` with `workspace: branch` so changes remain isolated until `critic` approves them against `main`.

## 3. Conventional Commit Guidelines
All commits must follow the Conventional Commits specification and reference the issue number when applicable:

| Type | When to Use | Example |
| :--- | :--- | :--- |
| `feat` | New solver capabilities, elements, or damage formulations | `feat(fe): add Q4 Helmholtz damage assembler (#14)` |
| `fix` | Bug fixes in C kernels, assembly, or Python driver | `fix(csrc): correct dmg pointer cast in python_module (#19)` |
| `perf` | Computational speedups in C quadrature or assembly | `perf(csrc): vectorize Q8 serendipity shape derivatives` |
| `test` | Adding or updating unit tests and verification benchmarks | `test(shape): add Kronecker delta tests for Q8 nodes` |
| `docs` | Documentation, docstrings, or TOML parameter guides | `docs(readme): document Taylor-Hood DOF ordering` |
| `refactor` | Code restructuring without behavioral changes | `refactor(assembly): streamline CSC matrix bincount map` |

## 4. Pre-Push & Pre-Merge Verification Checklist
Before pushing a branch or opening a pull request:
1. [ ] Recompile C extension (if `csrc/` or `setup.py` changed): `uv pip install -e . --no-build-isolation`
2. [ ] Passing unit tests: `uv run pytest -v`
3. [ ] Passing analytical benchmark: `uv run python .agents/skills/fem-verifier/scripts/verify_gravity_solve.py`
4. [ ] Clean lint and formatting: `uv run ruff check && uv run ruff format --check`
5. [ ] `critic` subagent verdict is `APPROVED` against `main`

## 5. Pushing & Opening a Pull Request
1. Push the feature branch to GitHub:
   ```bash
   git push -u origin HEAD
   ```
2. Open a pull request that auto-closes the issue on merge:
   ```bash
   gh pr create --fill --body "Closes #<ISSUE_NUMBER>"
   ```
