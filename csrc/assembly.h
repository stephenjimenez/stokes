/* Tools for assembling the system of equations. */

#ifndef STOKES_ASSEMBLY_H
#define STOKES_ASSEMBLY_H

#define SQRT_06 0.774596669241483377035853079956

static const double Q9_GAUSS_POINTS[9][2] = {
    {-SQRT_06, -SQRT_06}, {0.0, -SQRT_06}, {SQRT_06, -SQRT_06},
    {-SQRT_06, 0.0},      {0.0, 0.0},      {SQRT_06, 0.0},
    {-SQRT_06, SQRT_06},  {0.0, SQRT_06},  {SQRT_06, SQRT_06}};

static const double Q9_GAUSS_WEIGHTS[9] = {
    25.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0, 40.0 / 81.0, 64.0 / 81.0,
    40.0 / 81.0, 25.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0};

int assemble_stokes(const int (*cells)[8], const int (*dofs)[20],
                    const double (*x)[2], const double* u,
                    const double (*dmg)[9], const double bf[2],
                    const double b_glen, const double newton_alpha,
                    double (*ae)[16][16], double (*be)[4][16], double (*fe)[16],
                    int n_cells);

int assemble_stokes_cell(const double x[8][2], const double u[16],
                         const double dmg[9], const double bf[2],
                         const double b_glen, const double newton_alpha,
                         double ae[16][16], double be[4][16], double fe[16]);

#endif
