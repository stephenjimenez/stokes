/* Tools for computing stress and strain rate. */

#include "kinematics.h"

#include "shape.h"

/* Evaluate strain rates a Gauss point. */
int eval_strainrate(const int (*cells)[8], const int (*dofs)[20],
                    const double (*x)[2], const double* u, const double xi,
                    const double eta, double (*d)[3], int n_cells) {
  double dndxi[2][8], dndx[2][8];
  double b[3][16];
  double xe[8][2], ue[16];
  double jac[2][2], ijac[2][2];

  /* Evaluate shape function gradients. */
  q8_shape_gradients(xi, eta, dndxi);

  for (int c = 0; c < n_cells; ++c) {
    /* Pack coordinates and velocities. */
    for (int i = 0; i < 8; ++i) {
      int j = cells[c][i];
      xe[i][0] = x[j][0];
      xe[i][1] = x[j][1];
      ue[2 * i] = u[dofs[c][2 * i]];
      ue[2 * i + 1] = u[dofs[c][2 * i + 1]];
    }

    /* Compute the Jacobian matrix. */
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        double tmp = 0.0;
        for (int k = 0; k < 8; ++k) tmp += dndxi[i][k] * xe[k][j];
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

    /* Pack the B matrix. */
    for (int i = 0; i < 8; ++i) {
      const int i1 = 2 * i;
      const int i2 = 2 * i + 1;
      b[0][i1] = dndx[0][i];
      b[0][i2] = 0.0;
      b[1][i1] = 0.0;
      b[1][i2] = dndx[1][i];
      b[2][i1] = dndx[1][i];
      b[2][i2] = dndx[0][i];
    }

    /* Compute strain rate. */
    double d11 = 0.0;
    double d22 = 0.0;
    double g12 = 0.0;
    for (int i = 0; i < 16; ++i) {
      d11 += b[0][i] * ue[i];
      d22 += b[1][i] * ue[i];
      g12 += b[2][i] * ue[i];
    }
    d[c][0] = d11;
    d[c][1] = d22;
    d[c][2] = g12;
  }

  return 0;
}
