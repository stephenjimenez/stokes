---
name: fem-verifier
description: Sub-second analytical and numerical benchmark verification for finite element assembly and Stokes solver.
---

# FEM Verifier & Benchmark Skill

Use this skill when modifying finite element assembly, C numerical kernels, boundary condition implementations, or non-linear solver logic in `stokes`.

## Purpose
Simulations in `examples/` take many time steps and several minutes to run. This skill runs a fast (< 0.1 second) 1-step Stokes flow solve under gravity to verify:
1. Non-linear Picard/Newton iterations converge cleanly to tolerance.
2. Global algebraic residual $\| \mathbf{K} \mathbf{u}_{vp} - \mathbf{f} \| / \| \mathbf{f} \| < 10^{-6}$.
3. Pressure is compressive and positive throughout the interior ($p > 0$).
4. Free-slip boundary conditions and velocity spreading directions match continuum mechanics.

## Quick Execution
Execute the verification benchmark:

```bash
uv run python .agents/skills/fem-verifier/scripts/verify_gravity_solve.py
```

## Interpreting Results
- **Exit code 0 (`[PASS]`):** System assembly, boundary condition elimination, and non-linear solver are mathematically healthy.
- **Exit code 1 (`[FAIL]`):**
  - If solver fails to converge: check C kernel assembly tangents, sign of pressure block $-B^T$, or regularized strain rate $\dot{\varepsilon}_e$.
  - If relative residual fails: check Dirichlet boundary elimination masks in `apply_bcs_inplace`.
  - If pressure is negative: check coordinate orientation or gravity acceleration sign convention.
