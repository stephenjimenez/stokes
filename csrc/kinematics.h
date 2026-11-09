/* Tools for computing kinematics. */

#ifndef STOKES_KINEMATICS_H
#define STOKES_KINEMATICS_H

int eval_strainrate(const int (*cells)[8], const int (*dofs)[20],
                    const double (*x)[2], const double* u, const double xi,
                    const double eta, double (*d)[3], int n_cells);

#endif
