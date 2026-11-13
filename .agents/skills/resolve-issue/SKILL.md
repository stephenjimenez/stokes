---
name: resolve-issue
description: End-to-end workflow to read a GitHub issue via gh, checkout a branch, scout, architect, implement, test, critique, and push to GitHub.
---

# End-to-End GitHub Issue Resolution Workflow

Use this skill when asked to implement, fix, or resolve a GitHub issue by number or URL.

## 1. Read Issue & Checkout Branch
1. Fetch the full issue specification and discussion using the GitHub CLI:
   ```bash
   gh issue view <ISSUE_NUMBER> --comments
   ```
2. Ensure `main` is clean and up to date, then create a descriptive branch linked to the issue:
   ```bash
   git checkout main && git pull --ff-only
   git checkout -b <type>/issue-<ISSUE_NUMBER>-<short-slug>
   ```
   *(Use `feat/`, `fix/`, `perf/`, or `refactor/` matching Conventional Commits.)*

## 2. Scout & Architect (`@architect`)
1. Locate all affected files in `csrc/`, `src/stokes/`, and `tests/`.
2. For non-trivial math, physics, or C-kernel tasks, invoke the `architect` subagent to produce an implementation blueprint covering:
   - Weak forms, discretization, or DOF mappings.
   - Exact C99 signatures (`csrc/*.h`, `csrc/python_module.c`) and NumPy array shapes/contiguity.
   - Specific test cases to verify the change.

## 3. Implement (`@generator` + `c-kernel-tooling`)
1. Implement the changes following the Architect's specification.
2. If any file in `csrc/` or `setup.py` was modified, recompile and smoke-test using `c-kernel-tooling`:
   ```bash
   uv pip install -e . --no-build-isolation
   uv run python -c "import stokes._core; print('stokes._core loaded successfully')"
   ```
3. Format and lint the code:
   ```bash
   uv run ruff check --fix && uv run ruff format
   ```

## 4. Verify & Test (`@tester` + `fem-verifier`)
1. Invoke `tester` (or write unit tests in `tests/`) to cover the new functionality or bug fix.
2. Run the unit test suite and analytical FEM benchmark:
   ```bash
   uv run pytest -v
   uv run python .agents/skills/fem-verifier/scripts/verify_gravity_solve.py
   ```
3. If `csrc/` pointer logic was modified, run the ASan memory check from `c-kernel-tooling`.

## 5. Audit (`@critic`) & Document (`@docwriter`)
1. Invoke `critic` to review `git diff main` against mathematical, physical unit, and C-API invariants.
2. If `critic` returns `REJECTED`, fix all cited issues and re-run Step 4 before proceeding.
3. If user-facing parameters, TOML schemas, or public functions changed, invoke `docwriter` to update docstrings and documentation.

## 6. Commit & Push to GitHub (`git-workflow`)
1. Stage and commit changes using Conventional Commits, referencing the issue number:
   ```bash
   git add -A
   git commit -m "<type>(<scope>): <summary> (#<ISSUE_NUMBER>)"
   ```
2. Push the branch to origin:
   ```bash
   git push -u origin HEAD
   ```
3. (Optional, if requested) Open a pull request linking the issue:
   ```bash
   gh pr create --fill --body "Closes #<ISSUE_NUMBER>"
   ```
