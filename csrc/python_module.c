#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "assembly.h"
#include "kinematics.h"

/* Wrapper for assemble_stokes */
static PyObject* py_assemble_stokes(PyObject* self, PyObject* args) {
  PyArrayObject *py_cells, *py_dofs, *py_x, *py_u, *py_bf, *py_dmg;
  PyArrayObject *py_ae, *py_be, *py_fe;
  double b_glen, newton_alpha;

  if (!PyArg_ParseTuple(
          args, "O!O!O!O!O!O!ddO!O!O!", &PyArray_Type, &py_cells, &PyArray_Type,
          &py_dofs, &PyArray_Type, &py_x, &PyArray_Type, &py_u, &PyArray_Type,
          &py_dmg, &PyArray_Type, &py_bf, &b_glen, &newton_alpha, &PyArray_Type,
          &py_ae, &PyArray_Type, &py_be, &PyArray_Type, &py_fe)) {
    return NULL;
  }

  int n_cells = (int)PyArray_DIM(py_cells, 0);

  /* Extract direct raw C pointers from NumPy arrays (zero-copy) */
  const int (*cells)[8] = (const int (*)[8])PyArray_DATA(py_cells);
  const int (*dofs)[20] = (const int (*)[20])PyArray_DATA(py_dofs);
  const double (*x)[2] = (const double (*)[2])PyArray_DATA(py_x);
  const double* dmg = (const double*)PyArray_DATA(py_dmg);
  const double* u = (const double*)PyArray_DATA(py_u);
  const double* bf = (const double*)PyArray_DATA(py_bf);

  double (*ae)[16][16] = (double (*)[16][16])PyArray_DATA(py_ae);
  double (*be)[4][16] = (double (*)[4][16])PyArray_DATA(py_be);
  double (*fe)[16] = (double (*)[16])PyArray_DATA(py_fe);

  /* Release the Python Global Interpreter Lock (GIL) during C computation */
  int ierr = 0;
  Py_BEGIN_ALLOW_THREADS ierr = assemble_stokes(
      cells, dofs, x, u, dmg, bf, b_glen, newton_alpha, ae, be, fe, n_cells);
  Py_END_ALLOW_THREADS

      if (ierr != 0) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Stokes assembly failed: negative or zero Jacobian detected.");
    return NULL;
  }

  Py_RETURN_NONE;
}

/* Wrapper for eval_strainrate */
static PyObject* py_eval_strainrate(PyObject* self, PyObject* args) {
  PyArrayObject *py_cells, *py_dofs, *py_x, *py_u, *py_epsilon;
  double xi, eta;

  if (!PyArg_ParseTuple(args, "O!O!O!O!ddO!", &PyArray_Type, &py_cells,
                        &PyArray_Type, &py_dofs, &PyArray_Type, &py_x,
                        &PyArray_Type, &py_u, &xi, &eta, &PyArray_Type,
                        &py_epsilon)) {
    return NULL;
  }

  int n_cells = (int)PyArray_DIM(py_cells, 0);

  /* Extract direct raw C pointers from NumPy arrays (zero-copy) */
  const int (*cells)[8] = (const int (*)[8])PyArray_DATA(py_cells);
  const int (*dofs)[20] = (const int (*)[20])PyArray_DATA(py_dofs);
  const double (*x)[2] = (const double (*)[2])PyArray_DATA(py_x);
  const double* u = (const double*)PyArray_DATA(py_u);
  double (*epsilon)[3] = (double (*)[3])PyArray_DATA(py_epsilon);

  /* Release the GIL during C computation */
  int ierr = 0;
  Py_BEGIN_ALLOW_THREADS ierr =
      eval_strainrate(cells, dofs, x, u, xi, eta, epsilon, n_cells);
  Py_END_ALLOW_THREADS

      if (ierr != 0) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Strain rate evaluation failed: negative or zero Jacobian detected.");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyMethodDef CoreMethods[] = {
    {"assemble_stokes", py_assemble_stokes, METH_VARARGS,
     "Assembles 2D Stokes element matrices in parallel C."},
    {"eval_strainrate", py_eval_strainrate, METH_VARARGS,
     "Evaluates 2D strain rate tensor across cells at reference coordinates "
     "(xi, eta)."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef coremodule = {PyModuleDef_HEAD_INIT, "_core",
                                        "Compiled Stokes FE kernels", -1,
                                        CoreMethods};

PyMODINIT_FUNC PyInit__core(void) {
  import_array(); /* Required by NumPy C-API */
  return PyModule_Create(&coremodule);
}
