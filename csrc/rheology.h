/* Tools for rheology. */

#ifndef STOKES_RHEOLOGY_H
#define STOKES_RHEOLOGY_H

#include <math.h>

#define MIN_DEQ2 1e-14

/* Compute the Viscosity using Glen's law. */
static inline double glen_viscosity(double b_glen, double deq2) {
  return 0.5 * b_glen / cbrt(deq2 + MIN_DEQ2);
}

#endif
