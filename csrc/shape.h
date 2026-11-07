/* Finite element shape functions. */

#ifndef STOKES_SHAPE_H
#define STOKES_SHAPE_H

static inline void q4_shape_functions(double xi, double eta, double n[4]) {
  n[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
  n[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
  n[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
  n[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
}

static inline void q4_shape_gradients(double xi, double eta, double dn[2][4]) {
  dn[0][0] = -0.25 * (1.0 - eta);
  dn[0][1] = 0.25 * (1.0 - eta);
  dn[0][2] = 0.25 * (1.0 + eta);
  dn[0][3] = -0.25 * (1.0 + eta);
  dn[1][0] = -0.25 * (1.0 - xi);
  dn[1][1] = -0.25 * (1.0 + xi);
  dn[1][2] = 0.25 * (1.0 + xi);
  dn[1][3] = 0.25 * (1.0 - xi);
}

static inline void q8_shape_functions(double xi, double eta, double n[8]) {
  n[0] = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0);
  n[1] = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0);
  n[2] = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0);
  n[3] = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0);
  n[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta);
  n[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta);
  n[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta);
  n[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta);
}

static inline void q8_shape_gradients(double xi, double eta, double dn[2][8]) {
  dn[0][0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta);
  dn[0][1] = 0.25 * (1.0 - eta) * (2.0 * xi - eta);
  dn[0][2] = 0.25 * (1.0 + eta) * (2.0 * xi + eta);
  dn[0][3] = 0.25 * (1.0 + eta) * (2.0 * xi - eta);
  dn[0][4] = -xi * (1.0 - eta);
  dn[0][5] = 0.5 * (1.0 - eta * eta);
  dn[0][6] = -xi * (1.0 + eta);
  dn[0][7] = -0.5 * (1.0 - eta * eta);
  dn[1][0] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta);
  dn[1][1] = 0.25 * (1.0 + xi) * (-xi + 2.0 * eta);
  dn[1][2] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta);
  dn[1][3] = 0.25 * (1.0 - xi) * (-xi + 2.0 * eta);
  dn[1][4] = -0.5 * (1.0 - xi * xi);
  dn[1][5] = -eta * (1.0 + xi);
  dn[1][6] = 0.5 * (1.0 - xi * xi);
  dn[1][7] = -eta * (1.0 - xi);
}

#endif
