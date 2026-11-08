/* Tools for assembling the system of equations. */

#include "assembly.h"

#include "rheology.h"
#include "shape.h"

/* Assemble the Stokes system of equations. */
int assemble_stokes(const int (*cells)[8], const int (*dofs)[20],
                    const double (*x)[2], const double* u, const double bf[2],
                    const double b_glen, const double newton_alpha,
                    double (*ae)[16][16], double (*be)[4][16], double (*fe)[16],
                    int n_cells) {
  int global_err = 0;
  for (int i = 0; i < n_cells; ++i) {
    double ue[16], xe[8][2];
    if (global_err != 0) return 1;
    for (int j = 0; j < 8; ++j) {
      ue[2 * j] = u[dofs[i][2 * j]];
      ue[2 * j + 1] = u[dofs[i][2 * j + 1]];
      xe[j][0] = x[cells[i][j]][0];
      xe[j][1] = x[cells[i][j]][1];
    }
    int local_err = assemble_stokes_cell(xe, ue, bf, b_glen, newton_alpha,
                                         ae[i], be[i], fe[i]);
    if (local_err != 0) {
      global_err = local_err;
    }
  }
  return global_err;
}

/* Assemble matrices for a single element. */
int assemble_stokes_cell(const double x[8][2], const double u[16],
                         const double bf[2], const double b_glen,
                         const double newton_alpha, double ae[16][16],
                         double be[4][16], double fe[16]) {
  double n4[4], n8[8];
  double dndxi[2][8], dndx[2][8];
  double jac[2][2], ijac[2][2];
  double b[3][16], div[16];
  double h[16];
  for (int i = 0; i < 16; ++i) {
    fe[i] = 0.0;
    for (int j = 0; j < 16; ++j) ae[i][j] = 0.0;
    for (int j = 0; j < 4; ++j) be[j][i] = 0.0;
  }
  for (int gp = 0; gp < 9; ++gp) {
    const double xi = Q9_GAUSS_POINTS[gp][0];
    const double eta = Q9_GAUSS_POINTS[gp][1];
    const double w = Q9_GAUSS_WEIGHTS[gp];

    /* Evaluate shape functions. */
    q4_shape_functions(xi, eta, n4);
    q8_shape_functions(xi, eta, n8);
    q8_shape_gradients(xi, eta, dndxi);

    /* Compute the Jacobian matrix. */
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        double tmp = 0.0;
        for (int k = 0; k < 8; ++k) tmp += dndxi[i][k] * x[k][j];
        jac[i][j] = tmp;
      }
    }

    /* Invert the Jacobian. */
    const double det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    if (det <= 0.0) return 1;
    const double inv = 1.0 / det;
    ijac[0][0] = jac[1][1] * inv;
    ijac[0][1] = -jac[0][1] * inv;
    ijac[1][0] = -jac[1][0] * inv;
    ijac[1][1] = jac[0][0] * inv;

    /* Compute Cartesian shape function gradients. */
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 8; ++j)
        dndx[i][j] = ijac[i][0] * dndxi[0][j] + ijac[i][1] * dndxi[1][j];

    /* Pack the B matrix and divergence array. */
    for (int i = 0; i < 8; ++i) {
      const int i1 = 2 * i;
      const int i2 = 2 * i + 1;
      b[0][i1] = dndx[0][i];
      b[0][i2] = 0.0;
      b[1][i1] = 0.0;
      b[1][i2] = dndx[1][i];
      b[2][i1] = dndx[1][i];
      b[2][i2] = dndx[0][i];
      div[i1] = dndx[0][i];
      div[i2] = dndx[1][i];
    }

    /* Compute strain rate. */
    double e11 = 0.0;
    double e22 = 0.0;
    double g12 = 0.0;
    for (int i = 0; i < 16; ++i) {
      e11 += b[0][i] * u[i];
      e22 += b[1][i] * u[i];
      g12 += b[2][i] * u[i];
    }
    double e12 = 0.5 * g12;

    /* Compute the viscosity. */
    const double deq2 = 0.5 * (e11 * e11 + e22 * e22) + e12 * e12;
    const double mu = glen_viscosity(b_glen, deq2);

    /* Compute Newton tangent scalar gamma. */
    double gamma = 0.0;
    if (newton_alpha > 0.0) {
      gamma = newton_alpha * (2.0 / 3.0) * (mu / (deq2 + MIN_DEQ2));
      for (int i = 0; i < 16; ++i)
        h[i] = b[0][i] * e11 + b[1][i] * e22 + b[2][i] * e12;
    } else {
      for (int i = 0; i < 16; ++i) h[i] = 0.0;
    }

    /* Update the [A] matrix. */
    const double w_det = w * det;
    for (int i = 0; i < 16; ++i) {
      const double b0_i = b[0][i];
      const double b1_i = b[1][i];
      const double b2_i = b[2][i];
      const double h_i = h[i];
      for (int j = 0; j < 16; ++j) {
        const double picard =
            2.0 * (b0_i * b[0][j] + b1_i * b[1][j]) + b2_i * b[2][j];
        const double tangent = gamma * h_i * h[j];
        ae[i][j] += (mu * picard - tangent) * w_det;
      }
    }

    /* Update the [B] matrix. */
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 16; ++j) be[i][j] += n4[i] * div[j] * w_det;

    /* Update the {f} vector. */
    for (int i = 0; i < 8; ++i) {
      fe[2 * i] += bf[0] * n8[i] * w_det;
      fe[2 * i + 1] += bf[1] * n8[i] * w_det;
    }
    if (gamma > 0.0) {
      const double rhs_scale = 2.0 * deq2 * gamma * w_det;
      for (int i = 0; i < 16; ++i) {
        fe[i] -= rhs_scale * h[i];
      }
    }
  }
  return 0;
}
