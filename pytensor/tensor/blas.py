"""Ops and optimizations for using BLAS calls

BLAS = Basic Linear Algebra Subroutines
Learn more about BLAS here:
    http://www.netlib.org/blas/blast-forum/
The standard BLAS libraries implement what is called "legacy BLAS" in that
document.

This documentation describes PyTensor's BLAS optimization pipeline.

Where there is a discrepancy between how things do work and how they *should*
work, both aspects should be documented.

There are four kinds of BLAS Ops in PyTensor:
    - Python implementations (this file)
    - SciPy-based (blas_scipy)
    - C-based (blas_c)

Notes
-----
Unfortunately (because it's confusing) this file currently contains Ops
that contain both Python and C versions.  I think it would be better to
move the C implementations to blas_c so that this file is pure Python.
-JB


Ops
===

GEMM: Dot22, Dot22Scalar, GemmRelated, Gemm
-------------------------------------------

The BLAS GEMM operation implements Z <- a X Y + b Z,
where Z, X and Y are matrices, and a and b are scalars.

Dot22 is a GEMM where a=1, b=0, and Z is allocated every time.

Dot22Scalar is a GEMM where b=0 and Z is allocated every time.

Gemm is a GEMM in all its generality.

In the future we can refactor the GemmRelated, Gemm, Dot22 and
Dot22Scalar Ops into a single Op.  That new Op (Gemm2) is basically a
normal Gemm, but with an additional configuration variable that says
to ignore the input Z.  Setting that configuration variable to True
would make Gemm2 equivalent to the current Dot22 and Dot22Scalar.
This would make the file a lot easier to read, and save a few hundred
lines of library, to say nothing of testing and documentation.


GEMV: Gemv
----------

The BLAS GEMV operation implements Z <- a X Y + b Z,
where X is a matrix, Y, and Z are vectors, and a and b are scalars.


GER: Ger
--------

The BLAS GER operation implements Z <- a X' Y + Z,
where X and Y are vectors, and matrix Z gets a rank-1 update.


Other Notable BLAS-related Ops
------------------------------

SYRK is another useful special case of GEMM. Particularly SYRK preserves
symmetry in the matrix that it updates.  See how the linear-algebra module uses
symmetry hints before implementing this Op, so that this Op is compatible with
that system.


Optimizations
=============

The optimization pipeline works something like this:

    1. identify dot22 from dot
    2. identify gemm from dot22
    3. identify dot22scalar from dot22 that are not gemm
    4. specialize gemm to gemv where applicable
    5. specialize gemm to ger where applicable
    6. specialize dot22 -> gemv or ger where applicable

:note: GEMM is the most canonical BLAS signature that we deal with so far, it
    would be good to turn most things into GEMM (dot, inner, outer, dot22,
    dot22scalar), and then to specialize from gemm to the various other L2 and
    L3 operations.

Identify Dot22
--------------

Numpy's dot supports arguments that are of any rank, and we should support that
too (just for compatibility).  The BLAS optimizations work with Dot Ops whose
inputs are each either vector or matrix.  So the first part of the optimization
pipeline is to transform qualifying Dot Ops to Dot22 Ops. Dot22 Ops may be
transformed further, but they will get implemented by a BLAS call.

More precisely, Dot nodes whose inputs are all vectors or matrices and whose
inputs both have the same dtype, and whose dtype is float or complex, become
Dot22.  This is implemented in `local_dot_to_dot22`.


Identify Gemm from Dot22
------------------------

This is complicated, done in GemmOptimizer.

Identify Dot22Scalar from Dot22
-------------------------------

Dot22 Ops that remain after the GemmOptimizer is done have not
qualified as GEMM Ops. Still they might be scaled by a factor, in
which case we use Dot22Scalar which is like Gemm, but without the b
and the Z.  In the future it would be good to merge this into the
GemmOptimizer.

Specialize Gemm to Gemv
-----------------------

If arguments to GEMM are dimshuffled vectors, then we can use GEMV
instead. This optimization is `local_gemm_to_gemv`.

"""

import copy
import logging
import os
import time

import numpy as np


try:
    import numpy.__config__  # noqa
except ImportError:
    pass

from typing import Tuple, Union

import pytensor.scalar
from pytensor.compile.mode import optdb
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, view_roots
from pytensor.graph.features import ReplacementDidNotRemoveError, ReplaceValidate
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import (
    EquilibriumGraphRewriter,
    GraphRewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.graph.utils import InconsistencyError, MethodNotDefined, TestValueError
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.printing import FunctionPrinter, debugprint, pprint
from pytensor.scalar import bool as bool_t
from pytensor.tensor import basic as at
from pytensor.tensor.blas_headers import blas_header_text, blas_header_version
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import Dot, add, mul, neg, sub
from pytensor.tensor.rewriting.elemwise import local_dimshuffle_lift
from pytensor.tensor.shape import specify_broadcastable
from pytensor.tensor.type import (
    DenseTensorType,
    TensorType,
    integer_dtypes,
    tensor,
    values_eq_approx_remove_inf_nan,
)
from pytensor.utils import memoize


_logger = logging.getLogger("pytensor.tensor.blas")

try:
    import scipy.linalg.blas

    have_fblas = True
    try:
        fblas = scipy.linalg.blas.fblas
    except AttributeError:
        # A change merged in Scipy development version on 2012-12-02 replaced
        # `scipy.linalg.blas.fblas` with `scipy.linalg.blas`.
        # See http://github.com/scipy/scipy/pull/358
        fblas = scipy.linalg.blas
    _blas_gemv_fns = {
        np.dtype("float32"): fblas.sgemv,
        np.dtype("float64"): fblas.dgemv,
        np.dtype("complex64"): fblas.cgemv,
        np.dtype("complex128"): fblas.zgemv,
    }
except ImportError as e:
    have_fblas = False
    # This is used in Gemv and ScipyGer. We use CGemv and CGer
    # when config.blas__ldflags is defined. So we don't need a
    # warning in that case.
    if not config.blas__ldflags:
        _logger.warning(
            "Failed to import scipy.linalg.blas, and "
            "PyTensor flag blas__ldflags is empty. "
            "Falling back on slower implementations for "
            "dot(matrix, vector), dot(vector, matrix) and "
            f"dot(vector, vector) ({str(e)})"
        )


# If check_init_y() == True we need to initialize y when beta == 0.
def check_init_y():
    if check_init_y._result is None:
        if not have_fblas:
            check_init_y._result = False

        y = float("NaN") * np.ones((2,))
        x = np.ones((2,))
        A = np.ones((2, 2))
        gemv = _blas_gemv_fns[y.dtype]
        gemv(1.0, A.T, x, 0.0, y, overwrite_y=True, trans=True)
        check_init_y._result = np.isnan(y).any()

    return check_init_y._result


check_init_y._result = None


class Gemv(Op):
    """
    expression is beta * y + alpha * A x

    A is matrix
    x, y are vectors
    alpha, beta are scalars
    output is a vector that can be inplace on y

    """

    __props__ = ("inplace",)

    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}

    def __str__(self):
        if self.inplace:
            return "%s{inplace}" % self.__class__.__name__
        else:
            return "%s{no_inplace}" % self.__class__.__name__

    def make_node(self, y, alpha, A, x, beta):
        y = at.as_tensor_variable(y)
        x = at.as_tensor_variable(x)
        A = at.as_tensor_variable(A)
        alpha = at.as_tensor_variable(alpha)
        beta = at.as_tensor_variable(beta)
        if y.dtype != A.dtype or y.dtype != x.dtype:
            raise TypeError(
                "Gemv requires matching dtypes", (y.dtype, A.dtype, x.dtype)
            )
        if A.ndim != 2:
            raise TypeError("gemv requires matrix for A", A.type)
        if x.ndim != 1:
            raise TypeError("gemv requires vector for x", x.type)
        if y.ndim != 1:
            raise TypeError("gemv requires vector for y", y.type)

        inputs = [y, alpha, A, x, beta]

        if any(not isinstance(i.type, DenseTensorType) for i in inputs):
            raise NotImplementedError("Only dense tensor types are supported")

        return Apply(self, inputs, [y.type()])

    def perform(self, node, inputs, out_storage, params=None):
        y, alpha, A, x, beta = inputs
        if (
            have_fblas
            and y.shape[0] != 0
            and x.shape[0] != 0
            and y.dtype in _blas_gemv_fns
        ):
            gemv = _blas_gemv_fns[y.dtype]

            if A.shape[0] != y.shape[0] or A.shape[1] != x.shape[0]:
                raise ValueError(
                    "Incompatible shapes for gemv "
                    f"(beta * y + alpha * dot(A, x)). y: {y.shape}, A: {A.shape}, x: {x.shape}"
                )

            if beta == 0 and check_init_y():
                y.fill(0)

            # Here I suppose that A is in c order. If we don't make it
            #  explicitly as fortran order, scipy 0.7.2 seam to create
            #  a copy in fortran order instead of just reshaping it
            #  and using the trans flag.
            # If A is already in fortran order, make it in c order and using the
            #  trans flag don't seam to cause slowdown.
            # out_storage[0][0] = gemv(alpha, A, x, beta, y,
            #                         overwrite_y=self.inplace)
            out_storage[0][0] = gemv(
                alpha, A.T, x, beta, y, overwrite_y=self.inplace, trans=True
            )
        else:
            out = np.dot(A, x)
            if alpha != 1:
                out *= alpha
            if beta != 0:
                if beta != 1:
                    out += beta * y
                else:
                    out += y
            out_storage[0][0] = np.asarray(out, dtype=y.dtype)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


gemv_no_inplace = Gemv(inplace=False)
gemv_inplace = Gemv(inplace=True)
# For the user interface. Opt will make them inplace later
gemv = gemv_no_inplace


class Ger(Op):
    """
    BLAS defines general rank-1 update GER as A <- A + alpha x y'

    for matrix A, scalar alpha, vectors x and y.

    This interface to GER allows non-destructive operation on A via the
    `destructive` argument to the constructor.

    """

    __props__ = ("destructive",)

    def __init__(self, destructive):
        self.destructive = destructive
        if destructive:
            self.destroy_map = {0: [0]}

    def __str__(self):
        if self.destructive:
            return "%s{destructive}" % self.__class__.__name__
        else:
            return "%s{non-destructive}" % self.__class__.__name__

    def make_node(self, A, alpha, x, y):
        A = at.as_tensor_variable(A)
        y = at.as_tensor_variable(y)
        x = at.as_tensor_variable(x)
        alpha = at.as_tensor_variable(alpha)
        if not (A.dtype == x.dtype == y.dtype == alpha.dtype):
            raise TypeError(
                "ger requires matching dtypes", (A.dtype, alpha.dtype, x.dtype, y.dtype)
            )
        if alpha.ndim != 0:
            raise TypeError("ger requires scalar alpha", alpha.type)
        if A.ndim != 2:
            raise TypeError("ger requires matrix for A", A.type)
        if x.ndim != 1:
            raise TypeError("ger requires vector for x", x.type)
        if y.ndim != 1:
            raise TypeError("ger requires vector for y", y.type)

        if x.dtype not in ("float32", "float64", "complex64", "complex128"):
            raise TypeError("only float and complex types supported", x.dtype)

        inputs = [A, alpha, x, y]
        if any(not isinstance(i.type, DenseTensorType) for i in inputs):
            raise NotImplementedError("Only dense tensor types are supported")

        return Apply(self, inputs, [A.type()])

    def perform(self, node, inp, out, params=None):
        cA, calpha, cx, cy = inp
        (cZ,) = out
        if self.destructive:
            A = cA
        else:
            A = cA.copy()
        if calpha != 1:
            A += calpha * np.outer(cx, cy)
        else:
            A += np.outer(cx, cy)
        cZ[0] = A

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


ger = Ger(destructive=False)
ger_destructive = Ger(destructive=True)


def ldflags(libs=True, flags=False, libs_dir=False, include_dir=False):
    """Extract a list of compilation flags from config.blas__ldflags.

    Depending on the options, different type of flags will be kept.
    It returns a list of libraries against which an Op's object file
    should be linked to benefit from a BLAS implementation.

    Parameters
    ----------
    libs : bool, optional
        Extract flags starting with "-l" (the default is True).
    libs_dir : bool, optional
        Extract flags starting with "-L" (the default is False).
    include_dir : bool, optional
        Extract flags starting with "-I" (the default is False).
    flags: bool, optional
        Extract all the other flags (the default is False).

    Returns
    -------
    list of strings
        Extracted flags.

    """
    ldflags_str = config.blas__ldflags
    return _ldflags(
        ldflags_str=ldflags_str,
        libs=libs,
        flags=flags,
        libs_dir=libs_dir,
        include_dir=include_dir,
    )


@memoize
def _ldflags(ldflags_str, libs, flags, libs_dir, include_dir):
    """Extract list of compilation flags from a string.

    Depending on the options, different type of flags will be kept.

    Parameters
    ----------
    ldflags_str : string
        The string to process. Typically, this will be the content of
        `config.blas__ldflags`.
    libs : bool
        Extract flags starting with "-l".
    flags: bool
        Extract all the other flags.
    libs_dir: bool
        Extract flags starting with "-L".
    include_dir: bool
        Extract flags starting with "-I".

    Returns
    -------
    list of strings
        Extracted flags.

    """
    rval = []
    if libs_dir:
        found_dyn = False
        dirs = [x[2:] for x in ldflags_str.split() if x.startswith("-L")]
        l = _ldflags(
            ldflags_str=ldflags_str,
            libs=True,
            flags=False,
            libs_dir=False,
            include_dir=False,
        )
        for d in dirs:
            for f in os.listdir(d.strip('"')):
                if f.endswith(".so") or f.endswith(".dylib") or f.endswith(".dll"):
                    if any(f.find(ll) >= 0 for ll in l):
                        found_dyn = True
        if not found_dyn and dirs:
            _logger.warning(
                "We did not find a dynamic library in the "
                "library_dir of the library we use for blas. If you use "
                "ATLAS, make sure to compile it with dynamics library."
            )

    for t in ldflags_str.split():
        # Remove extra quote.
        if (t.startswith("'") and t.endswith("'")) or (
            t.startswith('"') and t.endswith('"')
        ):
            t = t[1:-1]

        try:
            t0, t1, t2 = t[0:3]
            assert t0 == "-"
        except Exception:
            raise ValueError(f'invalid token "{t}" in ldflags_str: "{ldflags_str}"')
        if libs_dir and t1 == "L":
            rval.append(t[2:])
        elif include_dir and t1 == "I":
            raise ValueError(
                "Include dirs are not used for blas. We disable"
                " this as this can hide other headers and this"
                " is not wanted.",
                t,
            )
            rval.append(t[2:])
        elif libs and t1 == "l":  # example -lmkl
            rval.append(t[2:])
        elif flags and t1 not in ("L", "I", "l"):  # example -openmp
            rval.append(t)
        elif flags and t1 == "L":
            # to find it when we load the compiled op if the env of the
            # used is not well configured.
            rval.append("-Wl,-rpath," + t[2:])
    return rval


class GemmRelated(COp):
    """Base class for Gemm and Dot22.

    This class provides a kind of templated gemm Op.

    """

    __props__: Union[Tuple, Tuple[str]] = ()

    def c_support_code(self, **kwargs):
        # return cblas_header_text()
        mod_str = """
        #ifndef MOD
        #define MOD %
        #endif
        static double time_time() // a time function like time.perf_counter()
        {
            struct timeval tv;
            gettimeofday(&tv, 0);
            return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
        }

        void compute_strides(npy_intp *shape, int N_shape, int type_size, npy_intp *res) {
            int s;
            res[N_shape - 1] = type_size;
            for (int i = N_shape - 1; i > 0; i--) {
                s = shape[i];
                res[i - 1] = res[i] * (s > 0 ? s : 1);
            }
        }
        """
        return blas_header_text() + mod_str

    def c_headers(self, **kwargs):
        # std.cout doesn't require the '%' symbol to print stuff...
        # so it works much better with python's string-substitution stuff.
        return ["<iostream>", "<time.h>", "<sys/time.h>"]

    def c_libraries(self, **kwargs):
        return ldflags()

    # code_cache_version is built by subclasses from
    # build_gemm_version

    def c_compile_args(self, **kwargs):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return ldflags(libs=False, include_dir=True)

    declare_NS = """
        int unit = 0;

        int type_num = PyArray_DESCR(%(_x)s)->type_num;
        int type_size = PyArray_DESCR(%(_x)s)->elsize; // in bytes

        npy_intp* Nx = PyArray_DIMS(%(_x)s);
        npy_intp* Ny = PyArray_DIMS(%(_y)s);
        npy_intp* Nz = 0; //PyArray_DIMS(%(_zout)s);

        npy_intp* Sx = PyArray_STRIDES(%(_x)s);
        npy_intp* Sy = PyArray_STRIDES(%(_y)s);
        npy_intp* Sz = 0; //PyArray_STRIDES(%(_zout)s);

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;
        """

    # implement if you don't have an inplace props
    # setup_z_Nz_Sz = None
    # otherwise implement
    # setup_z_Nz_Sz_inplace = None
    # setup_z_Nz_Sz_outplace = None

    check_xyz_rank2 = """
        if (PyArray_NDIM(%(_x)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(x) != 2. rank(x) is %%d.",
                         PyArray_NDIM(%(_x)s));
            %(fail)s;
        }
        if (PyArray_NDIM(%(_y)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(y) != 2. rank(y) is %%d.", PyArray_NDIM(%(_y)s));
            %(fail)s;
        }
        if (%(_zout)s && PyArray_NDIM(%(_zout)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(z) != 2. rank(z) is %%d.", PyArray_NDIM(%(_zout)s));
            %(fail)s;
        }
        """
    check_xyz_double_or_float = """
        if ((PyArray_DESCR(%(_x)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_x)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_y)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_y)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_zout)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_zout)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_x)s)->type_num != PyArray_DESCR(%(_y)s)->type_num)
            ||(PyArray_DESCR(%(_x)s)->type_num != PyArray_DESCR(%(_zout)s)->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(x), type(y), type(z) are not all the same"); %(fail)s; }
        """

    # it is not necessary that a or b have the same type as x,y,z
    check_ab_double_or_float = """
        if ((PyArray_DESCR(%(_a)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_a)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_b)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_b)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(b) is not double or float"); %(fail)s;}
        """

    # broadcast_xy = None

    check_dims = """
        if (Nx[0] !=1 && Nz[0] != 1 && Nx[0] != Nz[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld rows but z has %%ld rows",
                (long int)Nx[0], (long int)Nz[0]);
            %(fail)s;
        }
        if (Nx[1] != Ny[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld cols (and %%ld rows) but y has %%ld rows (and %%ld cols)",
                (long int)Nx[1], (long int)Nx[0], (long int)Ny[0], (long int)Ny[1]);
            %(fail)s;
        }
        if (Ny[1] != 1 && Nz[1]!= 1 && Ny[1] != Nz[1])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: y has %%ld cols but z has %%ld cols",
                (long int)Ny[1], (long int)Nz[1]);
            %(fail)s;
        }

        // We must not raise an error when Nx[1] == 0. This would disable cases
        // that numpy.dot accept.
        """

    check_strides = """
        /*
        If some matrices are not contiguous on either dimensions,
        or have invalid strides, copy their content into a contiguous one
        */
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
            || ((Sx[0] != type_size) && (Sx[1] != type_size)))
        {
            PyArrayObject * _x_copy = (PyArrayObject *) PyArray_Copy(%(_x)s);
            if (!_x_copy)
                %(fail)s
            Py_XDECREF(%(_x)s);
            %(_x)s = _x_copy;
            Sx = PyArray_STRIDES(%(_x)s);
            if ((Sx[0] < 1) || (Sx[1] < 1)) {
                compute_strides(Nx, 2, type_size, Sx);
            }
        }

        if ((Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
            || ((Sy[0] != type_size) && (Sy[1] != type_size)))
        {
            PyArrayObject * _y_copy = (PyArrayObject *) PyArray_Copy(%(_y)s);
            if (!_y_copy)
                %(fail)s
            Py_XDECREF(%(_y)s);
            %(_y)s = _y_copy;
            Sy = PyArray_STRIDES(%(_y)s);
            if ((Sy[0] < 1) || (Sy[1] < 1)) {
                compute_strides(Ny, 2, type_size, Sy);
            }
        }

        if ((Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size)
            || ((Sz[0] != type_size) && (Sz[1] != type_size)))
        {
            PyArrayObject * _z_copy = (PyArrayObject *) PyArray_Copy(%(_zout)s);
            if (!_z_copy)
                %(fail)s
            Py_XDECREF(%(_zout)s);
            %(_zout)s = _z_copy;
            Sz = PyArray_STRIDES(%(_zout)s);
            if ((Sz[0] < 1) || (Sz[1] < 1)) {
                compute_strides(Nz, 2, type_size, Sz);
            }
        }
        """

    encode_strides_in_unit = """
        /*
        encode the stride structure of _x,_y,_zout into a single integer
        */
        unit |= ((Sx[1] == type_size || Nx[1]==1) ? 0x0 : (Sx[0] == type_size || Nx[0]==1) ? 0x1 : 0x2) << 8;
        unit |= ((Sy[1] == type_size || Ny[1]==1) ? 0x0 : (Sy[0] == type_size || Ny[0]==1) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size || Nz[1]==1) ? 0x0 : (Sz[0] == type_size || Nz[0]==1) ? 0x1 : 0x2) << 0;
        """

    compute_strides = """
        /* create appropriate strides for malformed matrices that are row or column
         * vectors, or empty matrices.
         * In that case, the value of the stride does not really matter, but
         * some versions of BLAS insist that:
         *  - they are not smaller than the number of elements in the array,
         *  - they are not 0.
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : (Nx[1] + 1);
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : (Nx[0] + 1);
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : (Ny[1] + 1);
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : (Ny[0] + 1);
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : (Nz[1] + 1);
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : (Nz[0] + 1);
        """

    begin_switch_typenum = """
        switch (type_num)
        {
        """

    case_float = """
            case NPY_FLOAT:
            {
        """

    # case_float_ab_constants = None

    case_float_gemm = """
                float* x = (float*)PyArray_DATA(%(_x)s);
                float* y = (float*)PyArray_DATA(%(_y)s);
                float* z = (float*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                //double t0 = time_time();
                switch(unit)
                {
                    case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
                //fprintf(stderr, "Calling sgemm %%i %%i %%i %%i took %%f\\n", unit, Nz1, Nz0, Nx1, time_time() - t0);
        """

    case_double = """
            }
            break;
            case NPY_DOUBLE:
            {
        """

    # case_double_ab_constants = None

    case_double_gemm = """
                double* x = (double*)PyArray_DATA(%(_x)s);
                double* y = (double*)PyArray_DATA(%(_y)s);
                double* z = (double*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                //double t0 = time_time();
                //fprintf(stderr, "unit=%%x N= %%i %%i %%i S = %%i %%i %%i %%i %%i %%i\\n", unit,
                //Nz1, Nz0, Nx1,
                //sy_0, sy_1,
                //sx_0, sx_1,
                //sz_0, sz_1
                //);
                switch(unit)
                {
                    case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError,
                                             "some matrix has no unit stride");
                             %(fail)s;
                };
                //fprintf(stderr, "Calling dgemm %%i %%i %%i %%i took %%f\\n",
                //        unit, Nz1, Nz0, Nx1, time_time()- t0);
        """

    end_switch_typenum = """
            }
            break;
        }
        """

    def build_gemm_call(self):
        if hasattr(self, "inplace"):
            setup_z_Nz_Sz = "if(%(params)s->inplace){{{}}}else{{{}}}".format(
                self.setup_z_Nz_Sz_inplace,
                self.setup_z_Nz_Sz_outplace,
            )
        else:
            setup_z_Nz_Sz = self.setup_z_Nz_Sz

        return "".join(
            (
                self.declare_NS,
                self.check_xyz_rank2,
                setup_z_Nz_Sz,
                self.check_xyz_double_or_float,
                self.check_ab_double_or_float,
                self.broadcast_xy,
                self.check_dims,
                self.check_strides,
                self.encode_strides_in_unit,
                self.compute_strides,
                self.begin_switch_typenum,
                self.case_float,
                self.case_float_ab_constants,
                self.case_float_gemm,
                self.case_double,
                self.case_double_ab_constants,
                self.case_double_gemm,
                self.end_switch_typenum,
            )
        )

    def build_gemm_version(self):
        return (13, blas_header_version())


class Gemm(GemmRelated):
    """In-place version of matrix-matrix multiplication (with accumulation).

    When a and b are scalars and x, y, and z are matrices, then

        gemm(z,a,x,y,b)

    is similar to

        b*z + a*dot(x,y)

    The difference between the two is that the top form is destructive
    on z, whereas the bottom form is not.  Gemm works in-place on the
    storage associated with z, and the L{Variable} returned by Gemm
    has a storage that will be aliased to the storage of the z
    argument. Because of this in-place computation, an L{Apply} of
    this op will destroy the L{Variable} z on which it operates.  (See
    L{DestructiveOps} for an explanation of what destroying means in
    the context of pytensor graphs. See L{BlasLapackSupport} for more
    optimized linear algebra operations.)

    """

    E_rank = "gemm only works for rank 2"
    E_scalar = "gemm requires scalar argument"
    E_z_uniq = "argument z aliased to x or y"  # TODO: justify / delete this
    E_mixed = "gemm requires matching dtypes"
    E_float = "gemm requires floating-point dtypes"

    __props__ = ("inplace",)
    params_type = ParamsType(
        inplace=bool_t,
    )
    check_input = False

    def __init__(self, inplace):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __str__(self):
        if self.inplace:
            inplace_str = "inplace"
        else:
            inplace_str = "no_inplace"
        return f"{self.__class__.__name__}{{{inplace_str}}}"

    def __setstate__(self, dct):
        self.__dict__.update(dct)

        # Correctly reload older pickles where destroy_map were not
        # saved
        if "destroy_map" not in self.__dict__ and self.inplace:
            self.destroy_map = {0: [0]}

    def __getstate__(self):
        rval = self.__dict__.copy()
        # Do not serialize the setup code, it will be restored in __setstate__
        # depending on the value of 'inplace'
        rval.pop("setup_z_Nz_Sz", None)
        return rval

    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))

        if any(not isinstance(i.type, DenseTensorType) for i in inputs):
            raise NotImplementedError("Only dense tensor types are supported")

        if len(inputs) != 5:
            raise TypeError(
                f"Wrong number of inputs for {self} (expected 5, got {len(inputs)})"
            )
        z, a, x, y, b = inputs

        zr, xr, yr = [set(view_roots(i)) for i in (z, x, y)]

        # We want the gemm to be inplace. When this op is inplace, it
        # declare to be inplace only on z. So to make it safe, we
        # raise an error if z can be a view on x or y.

        # I don't know if PyTensor currently can support that case. As
        # this case don't happen in our code, I won't spent time
        # investigating this. So the assert is for safety.  I also
        # think there is another mechanism that would prevent this,
        # but I don't what to modify old code and have chance to break
        # something.
        if self.inplace:
            if zr.intersection(xr):
                raise InconsistencyError(Gemm.E_z_uniq, (z, x))
            if zr.intersection(yr):
                raise InconsistencyError(Gemm.E_z_uniq, (z, y))

        if z.ndim != 2:
            raise TypeError(Gemm.E_rank, z)
        if a.ndim != 0:
            raise TypeError(Gemm.E_scalar, a)
        if x.ndim != 2:
            raise TypeError(Gemm.E_rank, x)
        if y.ndim != 2:
            raise TypeError(Gemm.E_rank, y)
        if b.ndim != 0:
            raise TypeError(Gemm.E_scalar, b)

        if not (z.dtype == a.dtype == x.dtype == y.dtype == b.dtype):
            raise TypeError(Gemm.E_mixed, (z.dtype, a.dtype, x.dtype, y.dtype, b.dtype))

        if not z.dtype.startswith("float") and not z.dtype.startswith("complex"):
            raise TypeError(Gemm.E_float, (z.dtype))

        output = z.type()
        return Apply(self, inputs, [output])

    def perform(self, node, inp, out, params):
        z, a, x, y, b = inp
        (zout,) = out
        assert a.shape == ()
        assert b.shape == ()
        if not params.inplace:
            z = z.copy()  # the original z will not be changed
        if z.shape == ():
            z.itemset(z * a + b * np.dot(x, y))
            zout[0] = z
        else:
            # Broadcast Z if needed
            if (x.shape[0] > z.shape[0]) or (y.shape[1] > z.shape[1]):
                z = np.broadcast_to(
                    z, (max(x.shape[0], z.shape[0]), max(y.shape[1], z.shape[1]))
                ).copy()
            if b == 0.0:
                if a == 1.0:
                    z[:] = np.dot(x, y)
                elif a == -1.0:
                    z[:] = -np.dot(x, y)
                else:
                    z[:] = a * np.dot(x, y)
            elif b == 1.0:
                if a == 1.0:
                    z += np.dot(x, y)
                elif a == -1.0:
                    z -= np.dot(x, y)
                else:
                    z += a * np.dot(x, y)
            else:
                z *= b
                z += a * np.dot(x, y)
            zout[0] = z

    def infer_shape(self, fgraph, node, input_shapes):
        z_shape, _, x_shape, y_shape, _ = input_shapes
        return [
            (
                pytensor.scalar.scalar_maximum(z_shape[0], x_shape[0]),
                pytensor.scalar.scalar_maximum(z_shape[1], y_shape[1]),
            )
        ]

    setup_z_Nz_Sz_inplace = """
        // Needs broadcasting
        if (PyArray_DIMS(%(_z)s)[0] < Nx[0] || PyArray_DIMS(%(_z)s)[1] < Ny[1]){

            npy_intp dims[2];
            dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
            dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

            // Check if we need to allocate new array
            if((NULL == %(_zout)s)
                || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
                || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
            {
                // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
                Py_XDECREF(%(_zout)s);
                %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            }

            // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
            if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
            {
                %(fail)s;
            }

        } else {
            if (%(_zout)s != %(_z)s)
            {
                Py_XDECREF(%(_zout)s);
                %(_zout)s = %(_z)s;
                Py_INCREF(%(_zout)s);
            }
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

    setup_z_Nz_Sz_outplace = """
        npy_intp dims[2];
        dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
        dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

        // Check if we need to allocate new array
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
            || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_no_inplace output");
                %(fail)s
            }
        }

        // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
        if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
        {
            %(fail)s
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

    broadcast_xy = """
        // Broadcast X if needed
        if (Nz[0] > Nx[0])
        {
            npy_intp dims[2];
            dims[0] = Nz[0];
            dims[1] = Nx[1];
            // fprintf(stderr, "Gemm Broadcasting X into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *x_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!x_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_MoveInto(x_new, %(_x)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_x)s);
            %(_x)s = x_new;

            Nx = PyArray_DIMS(%(_x)s);
            Sx = PyArray_STRIDES(%(_x)s);
        }

        // Broadcast Y if needed
        if (Nz[1] > Ny[1])
        {
            npy_intp dims[2];
            dims[0] = Ny[0];
            dims[1] = Nz[1];
            // fprintf(stderr, "Gemm Broadcasting Y into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *y_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!y_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_MoveInto(y_new, %(_y)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_y)s);
            %(_y)s = y_new;

            Ny = PyArray_DIMS(%(_y)s);
            Sy = PyArray_STRIDES(%(_y)s);
        }

    """

    case_float_ab_constants = """
        #define REAL float
        float a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        float b = (PyArray_DESCR(%(_b)s)->type_num == NPY_FLOAT) ?
        (REAL)(((float*)PyArray_DATA(%(_b)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_b)s))[0]);
        #undef REAL
        """
    case_double_ab_constants = """
        #define REAL double
        double a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        double b = (PyArray_DESCR(%(_b)s)->type_num == NPY_FLOAT) ?
        (REAL)(((float*)PyArray_DATA(%(_b)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_b)s))[0]);
        #undef REAL
        """

    def c_code(self, node, name, inp, out, sub):
        _z, _a, _x, _y, _b = inp
        (_zout,) = out
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (7,) + gv
        else:
            return gv


gemm_inplace = Gemm(inplace=True)
gemm_no_inplace = Gemm(inplace=False)
# For the user interface. PyTensor optimization will make them inplace
gemm = gemm_no_inplace
pprint.assign(gemm_inplace, FunctionPrinter(["gemm_inplace"]))
pprint.assign(gemm_no_inplace, FunctionPrinter(["gemm_no_inplace"]))


def res_is_a(fgraph, var, op, maxclients=None):
    if maxclients is not None and var in fgraph.clients:
        retval = len(fgraph.get_clients(var)) <= maxclients
    else:
        retval = True

    return var.owner and var.owner.op == op and retval


def _as_scalar(res, dtype=None):
    """Return ``None`` or a `TensorVariable` of float type"""
    if dtype is None:
        dtype = config.floatX
    if all(s == 1 for s in res.type.shape):
        while res.owner and isinstance(res.owner.op, DimShuffle):
            res = res.owner.inputs[0]
        # may still have some number of True's
        if res.type.ndim > 0:
            rval = res.dimshuffle()
        else:
            rval = res
        if rval.type.dtype in integer_dtypes:
            # We check that the upcast of res and dtype won't change dtype.
            # If dtype is float64, we will cast int64 to float64.
            # This is valid when res is a scalar used as input to a dot22
            # as the cast of the scalar can be done before or after the dot22
            # and this will give the same result.
            if pytensor.scalar.upcast(res.dtype, dtype) == dtype:
                return at.cast(rval, dtype)
            else:
                return None

        return rval


def _is_real_matrix(res):
    return (
        res.type.dtype in ("float16", "float32", "float64")
        and res.type.ndim == 2
        and res.type.shape[0] != 1
        and res.type.shape[1] != 1
    )  # cope with tuple vs. list


def _is_real_vector(res):
    return (
        res.type.dtype in ("float16", "float32", "float64")
        and res.type.ndim == 1
        and res.type.shape[0] != 1
    )


def _beta_L_plus_alpha_M(fgraph, beta, L, alpha, M, recurse_flip=True):
    # print 'BETA L + ALPHA M', beta, L, alpha, M, recurse_flip
    # EXPRESSION: (beta * L) + (alpha * M)

    # we've already checked the client counts, now just make the type check.
    # if res_is_a(M, _dot22, 1):
    if M.owner and M.owner.op == _dot22:
        Ml, Mr = M.owner.inputs
        rval = [gemm_no_inplace(L, alpha, Ml, Mr, beta)]
        return rval, M

    # it also might be the case that there is a dimshuffle between the +
    # and the dot22. local_dot_to_dot22 in particular will put in such things.
    if (
        M.owner
        and isinstance(M.owner.op, DimShuffle)
        and M.owner.inputs[0].owner
        and isinstance(M.owner.inputs[0].owner.op, Dot22)
    ):
        MM = M.owner.inputs[0]
        if M.owner.op.new_order == (0,):
            # it is making a column MM into a vector
            MMl, MMr = MM.owner.inputs
            g = gemm_no_inplace(L.dimshuffle(0, "x"), alpha, MMl, MMr, beta)
            rval = [g.dimshuffle(0)]
            return rval, MM
        if M.owner.op.new_order == (1,):
            # it is making a row MM into a vector
            MMl, MMr = MM.owner.inputs
            g = gemm_no_inplace(L.dimshuffle("x", 0), alpha, MMl, MMr, beta)
            rval = [g.dimshuffle(1)]
            return rval, MM
        if len(M.owner.op.new_order) == 0:
            # it is making a row MM into a vector
            MMl, MMr = MM.owner.inputs
            g = gemm_no_inplace(L.dimshuffle("x", "x"), alpha, MMl, MMr, beta)
            rval = [g.dimshuffle()]
            return rval, MM

    if recurse_flip:
        return _beta_L_plus_alpha_M(fgraph, alpha, M, beta, L, recurse_flip=False)
    else:
        return False, False


def _gemm_canonicalize(fgraph, r, scale, rval, maxclients):
    # Tries to interpret node as a sum of scalars * (vectors or matrices)
    def scaled(thing):
        if scale == 1:
            return thing
        if scale == -1 and thing.type.dtype != "bool":
            return -thing
        else:
            return scale * thing

    if not isinstance(r.type, TensorType):
        return None

    if (r.type.ndim not in (1, 2)) or r.type.dtype not in (
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        rval.append(scaled(r))
        return rval

    if maxclients and len(fgraph.clients[r]) > maxclients:
        rval.append((scale, r))
        return rval

    if r.owner and r.owner.op == sub:
        _gemm_canonicalize(fgraph, r.owner.inputs[0], scale, rval, 1)
        _gemm_canonicalize(fgraph, r.owner.inputs[1], -scale, rval, 1)

    elif r.owner and r.owner.op == add:
        for i in r.owner.inputs:
            _gemm_canonicalize(fgraph, i, scale, rval, 1)

    elif r.owner and r.owner.op == neg:
        _gemm_canonicalize(fgraph, r.owner.inputs[0], -scale, rval, 1)

    elif r.owner and r.owner.op == mul:
        scalars = []
        vectors = []
        matrices = []
        for i in r.owner.inputs:
            if all(s == 1 for s in i.type.shape):
                while i.owner and isinstance(i.owner.op, DimShuffle):
                    i = i.owner.inputs[0]
                if i.type.ndim > 0:
                    scalars.append(i.dimshuffle())
                else:
                    scalars.append(i)
            elif _is_real_vector(i):
                vectors.append(i)
            elif _is_real_matrix(i):
                matrices.append(i)
            else:
                # just put the original arguments as in the base case
                rval.append((scale, r))
                return rval
        if len(matrices) == 1:
            assert len(vectors) == 0
            m = matrices[0]
            if len(scalars) == 0:
                _gemm_canonicalize(fgraph, m, scale, rval, 1)
            elif len(scalars) == 1:
                _gemm_canonicalize(fgraph, m, scaled(scalars[0]), rval, 1)
            else:
                _gemm_canonicalize(
                    fgraph, m, mul(scaled(scalars[0]), *scalars[1:]), rval, 1
                )
        elif len(vectors) == 1:
            assert len(matrices) == 0
            v = vectors[0]
            if len(scalars) == 0:
                _gemm_canonicalize(fgraph, v, scale, rval, 1)
            elif len(scalars) == 1:
                _gemm_canonicalize(fgraph, v, scaled(scalars[0]), rval, 1)
            else:
                _gemm_canonicalize(
                    fgraph, v, mul(scaled(scalars[0]), *scalars[1:]), rval, 1
                )
        else:  # lets not open this up
            rval.append((scale, r))
    else:
        rval.append((scale, r))
    return rval


def _factor_canonicalized(lst):
    # remove duplicates from canonicalized list

    # we only delete out of the right end of the list,
    # once i has touched a list element, it is permantent
    lst = list(lst)
    # print 'FACTOR', lst
    # for t in lst:
    #    if not isinstance(t, (list, tuple)):
    #        t = (t,)
    #    for e in t:
    #        try:
    #            pytensor.printing.debugprint(e)
    #        except TypeError:
    #            print e, type(e)
    i = 0
    while i < len(lst) - 1:
        try:
            s_i, M_i = lst[i]
        except Exception:
            i += 1
            continue

        j = i + 1
        while j < len(lst):
            try:
                s_j, M_j = lst[j]
            except Exception:
                j += 1
                continue

            if M_i is M_j:
                s_i = s_i + s_j
                lst[i] = (s_i, M_i)
                del lst[j]
            else:
                j += 1
        i += 1
    return lst


def _gemm_from_factored_list(fgraph, lst):
    """
    Returns None, or a list to replace node.outputs.

    """
    lst2 = []
    # Remove the tuple that can't be cast correctly.
    # This can happen when we try to cast a complex to a real
    for sM in lst:
        # Make every pair in list have matching dtypes
        # sM can be a tuple of 2 elements or an PyTensor variable.
        if isinstance(sM, tuple):
            sm0, sm1 = sM
            sm0 = at.as_tensor_variable(sm0)
            if pytensor.scalar.upcast(sm0.dtype, sm1.dtype) == sm1.dtype:
                lst2.append((at.cast(sm0, sm1.dtype), sM[1]))

    lst = lst2

    def item_to_var(t):
        try:
            s, M = t
        except Exception:
            return t
        if s == 1:
            return M
        if s == -1:
            return -M
        return s * M

    # Try every pair in the sM_list, trying to turn it into a gemm operation
    for i in range(len(lst) - 1):
        s_i, M_i = lst[i]

        for j in range(i + 1, len(lst)):
            s_j, M_j = lst[j]

            if not M_j.type.in_same_class(M_i.type):
                continue

            # print 'TRYING', (s_i, M_i, s_j, M_j)

            gemm_of_sM_list, old_dot22 = _beta_L_plus_alpha_M(
                fgraph, s_i, M_i, s_j, M_j
            )
            # print 'GOT IT', gemm_of_sM_list
            if gemm_of_sM_list:

                assert len(gemm_of_sM_list) == 1
                add_inputs = [
                    item_to_var(input) for k, input in enumerate(lst) if k not in (i, j)
                ]
                add_inputs.extend(gemm_of_sM_list)
                if len(add_inputs) > 1:
                    rval = [add(*add_inputs)]
                else:
                    rval = add_inputs
                # print "RETURNING GEMM THING", rval
                return rval, old_dot22


def _gemm_from_node2(fgraph, node):
    """

    TODO: In many expressions, there are many ways to turn it into a
    gemm.  For example dot(a,b) + c + d.  This function should return all
    of them, so that if one version of gemm causes a cycle in the graph, then
    another application of gemm can be tried.

    """
    lst = []
    t0 = time.perf_counter()
    _gemm_canonicalize(fgraph, node.outputs[0], 1.0, lst, 0)
    t1 = time.perf_counter()

    if len(lst) > 1:
        lst = _factor_canonicalized(lst)
        t2 = time.perf_counter()
        rval = _gemm_from_factored_list(fgraph, lst)
        t3 = time.perf_counter()

        # It can happen that _factor_canonicalized and
        # _gemm_from_factored_list return a node with an incorrect
        # type.  This happens in particular when one of the scalar
        # factors forces the upcast of the whole expression.  In that
        # case, we simply skip that candidate for Gemm.  This was
        # discussed in
        # http://groups.google.com/group/theano-dev/browse_thread/thread/a3096c82856e3ad5,
        # but never made it into a trac ticket.

        if rval and rval[0][0].type.in_same_class(node.outputs[0].type):
            return rval, t1 - t0, t2 - t1, t3 - t2

    return None, t1 - t0, 0, 0


class GemmOptimizer(GraphRewriter):
    """Graph optimizer for inserting Gemm operations."""

    def __init__(self):
        super().__init__()
        self.warned = False

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    def apply(self, fgraph):
        did_something = True
        nb_iter = 0
        nb_replacement = 0
        nb_replacement_didn_t_remove = 0
        nb_inconsistency_make = 0
        nb_inconsistency_replace = 0
        time_canonicalize = 0
        time_factor_can = 0
        time_factor_list = 0
        time_toposort = 0
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time

        def on_import(new_node):
            if new_node is not node:
                nodelist.append(new_node)

        u = pytensor.graph.rewriting.basic.DispatchingFeature(
            on_import, None, None, name="GemmOptimizer"
        )
        fgraph.attach_feature(u)
        while did_something:
            nb_iter += 1
            t0 = time.perf_counter()
            nodelist = pytensor.graph.basic.io_toposort(fgraph.inputs, fgraph.outputs)
            time_toposort += time.perf_counter() - t0
            did_something = False
            nodelist.reverse()
            for node in nodelist:
                if not (
                    isinstance(node.op, Elemwise)
                    and isinstance(
                        node.op.scalar_op,
                        (
                            pytensor.scalar.Add,
                            pytensor.scalar.Sub,
                            pytensor.scalar.Neg,
                            pytensor.scalar.Mul,
                        ),
                    )
                ):
                    continue
                if node not in fgraph.apply_nodes:
                    # This mean that we already removed this node from
                    # the graph
                    continue
                try:
                    new_outputs, time1, time2, time3 = _gemm_from_node2(fgraph, node)
                    time_canonicalize += time1
                    time_factor_can += time2
                    time_factor_list += time3
                except InconsistencyError:
                    nb_inconsistency_make += 1
                    continue
                if new_outputs:
                    new_outputs, old_dot22 = new_outputs
                    assert len(new_outputs) == len(node.outputs)
                    new_outputs[
                        0
                    ].tag.values_eq_approx = values_eq_approx_remove_inf_nan
                    try:
                        fgraph.replace_all_validate_remove(
                            list(zip(node.outputs, new_outputs)),
                            [old_dot22],
                            reason="GemmOptimizer",
                            # For now we disable the warning as we know case
                            # that we need to fix.
                            warn=False,  # warn=not self.warned
                        )
                        did_something = True
                        nb_replacement += 1
                    except InconsistencyError:
                        # TODO: retry other applications of gemm (see comment
                        # in _gemm_from_node)
                        nb_inconsistency_replace += 1
                    except ReplacementDidNotRemoveError:
                        nb_replacement_didn_t_remove += 1
                        self.warned = True
        fgraph.remove_feature(u)
        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    callbacks_time[k] = v - callbacks_before[k]
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}

        return (
            self,
            nb_iter,
            nb_replacement,
            nb_replacement_didn_t_remove,
            nb_inconsistency_make,
            nb_inconsistency_replace,
            time_canonicalize,
            time_factor_can,
            time_factor_list,
            time_toposort,
            validate_time,
            callback_time,
            callbacks_time,
        )

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, file=stream)
        print(blanc, " nb_iter", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_replacement_didn_t_remove", prof[3], file=stream)
        print(blanc, " nb_inconsistency_make", prof[4], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[5], file=stream)
        print(blanc, " time_canonicalize", prof[6], file=stream)
        print(blanc, " time_factor_can", prof[7], file=stream)
        print(blanc, " time_factor_list", prof[8], file=stream)
        print(blanc, " time_toposort", prof[9], file=stream)
        print(blanc, " validate_time", prof[10], file=stream)
        print(blanc, " callback_time", prof[11], file=stream)
        if prof[11] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[12].items(), key=lambda a: a[1]):
                if i[1] > 0:
                    print(i)


class Dot22(GemmRelated):
    """Compute a matrix-matrix product.

    This is a specialization of the more general Dot().

    """

    check_input = False

    def make_node(self, x, y):
        x = at.as_tensor_variable(x)
        y = at.as_tensor_variable(y)

        if any(not isinstance(i.type, DenseTensorType) for i in (x, y)):
            raise NotImplementedError("Only dense tensor types are supported")

        dtypes = ("float16", "float32", "float64", "complex64", "complex128")
        if x.type.ndim != 2 or x.type.dtype not in dtypes:
            raise TypeError(x)
        if y.type.ndim != 2 or y.type.dtype not in dtypes:
            raise TypeError(y)
        if y.type.dtype != x.type.dtype:
            raise TypeError("dtype mismatch to Dot22")
        outputs = [tensor(x.type.dtype, shape=(x.type.shape[0], y.type.shape[1]))]
        return Apply(self, [x, y], outputs)

    def perform(self, node, inp, out):
        x, y = inp
        (z,) = out
        try:
            z[0] = np.asarray(np.dot(x, y))
        except ValueError as e:
            # The error raised by numpy has no shape information, we mean to
            # add that
            e.args = e.args + (x.shape, y.shape)
            raise

    def infer_shape(self, fgraph, node, input_shapes):
        return [[input_shapes[0][0], input_shapes[1][1]]]

    setup_z_Nz_Sz = """
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_x)s)[0])
            || (PyArray_DIMS(%(_zout)s)[1] != PyArray_DIMS(%(_y)s)[1]))
        {
            if (NULL != %(_zout)s) Py_XDECREF(%(_zout)s);
            npy_intp dims[2];
            dims[0] = PyArray_DIMS(%(_x)s)[0];
            dims[1] = PyArray_DIMS(%(_y)s)[1];
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims,
                            PyArray_TYPE(%(_x)s));
            //fprintf(stderr, "Dot Allocating %%i %%i\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc dot22 output");
                %(fail)s
            }
        }
        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);

        """
    broadcast_xy = ""
    check_ab_double_or_float = ""
    case_float_ab_constants = """
                float a = 1.0;
                float b = 0.0;
        """
    case_double_ab_constants = """
                double a = 1.0;
                double b = 0.0;
        """

    def c_code(self, node, name, inp, out, sub):  # DEBUG
        _x, _y = inp
        (_zout,) = out
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        if len(self.c_libraries()) <= 0:
            raise NotImplementedError()
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (2,) + gv
        else:
            return gv


_dot22 = Dot22()


@node_rewriter([Dot])
def local_dot_to_dot22(fgraph, node):
    # This works for tensor.outer too because basic.outer is a macro that
    # produces a dot(dimshuffle,dimshuffle) of form 4 below
    if not isinstance(node.op, Dot):
        return

    if any(not isinstance(i.type, DenseTensorType) for i in node.inputs):
        return False

    x, y = node.inputs
    if y.type.dtype != x.type.dtype:
        # TODO: upcast one so the types match
        _logger.info(f"Not optimizing dot with inputs {x} {y} {x.type} {y.type}")
        return

    if y.type.dtype in ("float16", "float32", "float64", "complex64", "complex128"):
        if x.ndim == 2 and y.ndim == 2:
            new_out = [_dot22(*node.inputs)]
        elif x.ndim == 2 and y.ndim == 1:
            new_out = [_dot22(x, y.dimshuffle(0, "x")).dimshuffle(0)]
        elif x.ndim == 1 and y.ndim == 2:
            new_out = [_dot22(x.dimshuffle("x", 0), y).dimshuffle(1)]
        elif x.ndim == 1 and y.ndim == 1:
            new_out = [_dot22(x.dimshuffle("x", 0), y.dimshuffle(0, "x")).dimshuffle()]
        else:
            return
        copy_stack_trace(node.outputs, new_out)
        return new_out

    _logger.info(f"Not optimizing dot with inputs {x} {y} {x.type} {y.type}")


@node_rewriter([gemm_no_inplace], inplace=True)
def local_inplace_gemm(fgraph, node):
    if node.op == gemm_no_inplace:
        new_out = [gemm_inplace(*node.inputs)]
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([gemv_no_inplace], inplace=True)
def local_inplace_gemv(fgraph, node):
    if node.op == gemv_no_inplace:
        new_out = [gemv_inplace(*node.inputs)]
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([ger], inplace=True)
def local_inplace_ger(fgraph, node):
    if node.op == ger:
        new_out = [ger_destructive(*node.inputs)]
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([gemm_no_inplace])
def local_gemm_to_gemv(fgraph, node):
    """GEMM acting on row or column matrices -> GEMV."""
    if node.op == gemm_no_inplace:
        z, a, x, y, b = node.inputs
        if z.broadcastable == x.broadcastable == (True, False):
            r = gemv_no_inplace(z.dimshuffle(1), a, y.T, x.dimshuffle(1), b)
            new_out = [r.dimshuffle("x", 0)]
        elif z.broadcastable == y.broadcastable == (False, True):
            r = gemv_no_inplace(z.dimshuffle(0), a, x, y.dimshuffle(0), b)
            new_out = [r.dimshuffle(0, "x")]
        else:
            return
        copy_stack_trace(node.outputs, new_out)
        return new_out


@node_rewriter([gemm_no_inplace])
def local_gemm_to_ger(fgraph, node):
    """GEMM computing an outer-product -> GER."""
    if node.op == gemm_no_inplace:
        z, a, x, y, b = node.inputs
        if x.broadcastable[1] and y.broadcastable[0]:
            # x and y are both vectors so this might qualifies for a GER
            xv = x.dimshuffle(0)
            yv = y.dimshuffle(1)
            try:
                bval = at.get_scalar_constant_value(b)
            except NotScalarConstantError:
                # b isn't a constant, GEMM is doing useful pre-scaling
                return

            if bval == 1:  # best case a natural GER
                rval = ger(z, a, xv, yv)
                new_out = [rval]
            elif bval == 0:  # GER on zeros_like should be faster than GEMM
                zeros = at.zeros([x.shape[0], y.shape[1]], x.dtype)
                rval = ger(zeros, a, xv, yv)
                new_out = [rval]
            else:
                # if bval is another constant, then z is being usefully
                # pre-scaled and GER isn't really the right tool for the job.
                return
            copy_stack_trace(node.outputs, new_out)
            return new_out


# TODO: delete this optimization when we have the proper dot->gemm->ger pipeline
#      working
@node_rewriter([_dot22])
def local_dot22_to_ger_or_gemv(fgraph, node):
    """dot22 computing an outer-product -> GER."""
    if node.op == _dot22:
        x, y = node.inputs
        xb = x.broadcastable
        yb = y.broadcastable
        one = at.as_tensor_variable(np.asarray(1, dtype=x.dtype))
        zero = at.as_tensor_variable(np.asarray(0, dtype=x.dtype))
        if xb[1] and yb[0]:
            # x and y are both vectors so this might qualifies for a GER
            xv = x.dimshuffle(0)
            yv = y.dimshuffle(1)
            zeros = at.zeros([x.shape[0], y.shape[1]], dtype=x.dtype)
            rval = ger(zeros, one, xv, yv)
            new_out = [rval]
        elif xb[0] and yb[1]:
            # x and y are both vectors so this qualifies for a sdot / ddot
            # TODO: PyTensor doesn't have a sdot, but gemv is better than _dot22
            xv = x.dimshuffle(1)
            zeros = at.AllocEmpty(x.dtype)(1)
            rval = gemv_no_inplace(zeros, one, y.T, xv, zero)
            new_out = [rval.dimshuffle("x", 0)]
        elif xb[0] and not yb[0] and not yb[1]:
            # x is vector, y is matrix so try gemv
            xv = x.dimshuffle(1)
            zeros = at.AllocEmpty(x.dtype)(y.shape[1])
            rval = gemv_no_inplace(zeros, one, y.T, xv, zero)
            new_out = [rval.dimshuffle("x", 0)]
        elif not xb[0] and not xb[1] and yb[1]:
            # x is matrix, y is vector, try gemv
            yv = y.dimshuffle(0)
            zeros = at.AllocEmpty(x.dtype)(x.shape[0])
            rval = gemv_no_inplace(zeros, one, x, yv, zero)
            new_out = [rval.dimshuffle(0, "x")]
        else:
            return
        copy_stack_trace(node.outputs, new_out)
        return new_out


#################################
#
# Set up the BlasOpt optimizer
#
#################################

blas_optdb = SequenceDB()

# run after numerical stability optimizations (1.5)
optdb.register("BlasOpt", blas_optdb, "fast_run", "fast_compile", position=1.7)
# run before specialize (2.0) because specialize is basically a
# free-for-all that makes the graph crazy.

# fast_compile is needed to have GpuDot22 created.
blas_optdb.register(
    "local_dot_to_dot22",
    in2out(local_dot_to_dot22),
    "fast_run",
    "fast_compile",
    position=0,
)
blas_optdb.register("gemm_optimizer", GemmOptimizer(), "fast_run", position=10)
blas_optdb.register(
    "local_gemm_to_gemv",
    EquilibriumGraphRewriter(
        [
            local_gemm_to_gemv,
            local_gemm_to_ger,
            local_dot22_to_ger_or_gemv,
            local_dimshuffle_lift,
        ],
        max_use_ratio=5,
        ignore_newtrees=False,
    ),
    "fast_run",
    position=15,
)


# After destroyhandler(49.5) but before we try to make elemwise things
# inplace (75)
blas_opt_inplace = in2out(
    local_inplace_gemm, local_inplace_gemv, local_inplace_ger, name="blas_opt_inplace"
)
optdb.register(
    "InplaceBlasOpt",
    blas_opt_inplace,
    "fast_run",
    "inplace",
    "blas_opt_inplace",
    position=70.0,
)


class Dot22Scalar(GemmRelated):
    """Compute a matrix-matrix product.

    This is a specialization of the more general Dot()
    Used to call optimized gemm implementation.
    Also used to generate a gemm later.
    compute scalar*dot(x,y).

    """

    check_input = False

    def make_node(self, x, y, a):

        if any(not isinstance(i.type, DenseTensorType) for i in (x, y, a)):
            raise NotImplementedError("Only dense tensor types are supported")

        if a.ndim != 0:
            raise TypeError(Gemm.E_scalar, a)
        if x.ndim != 2:
            raise TypeError(Gemm.E_rank, x)
        if y.ndim != 2:
            raise TypeError(Gemm.E_rank, y)

        if not (a.dtype == x.dtype == y.dtype):
            raise TypeError(
                "Dot22Scalar requires matching dtypes", (a.dtype, x.dtype, y.dtype)
            )

        if not a.dtype.startswith("float") and not a.dtype.startswith("complex"):
            raise TypeError("Dot22Scalar requires float or complex args", a.dtype)

        sz = (x.type.shape[0], y.type.shape[1])
        outputs = [tensor(x.type.dtype, shape=sz)]
        return Apply(self, [x, y, a], outputs)

    def perform(self, node, inp, out):
        x, y, scalar = inp
        (z,) = out
        try:
            z[0] = np.asarray(scalar * np.dot(x, y))
        except ValueError as e:
            # The error raised by numpy has no shape information, we
            # mean to add that
            e.args = e.args + (x.shape, y.shape)
            raise

    def infer_shape(self, fgraph, node, input_shapes):
        return [[input_shapes[0][0], input_shapes[1][1]]]

    setup_z_Nz_Sz = Dot22.setup_z_Nz_Sz
    broadcast_xy = ""

    check_ab_double_or_float = """
        if ((PyArray_DESCR(%(_a)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_a)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError,
                         "type(a) is not double or float"); %(fail)s;}

        """
    case_float_ab_constants = """
        #define REAL float
        float a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        #undef REAL
        float b = 0.0;
        """

    case_double_ab_constants = """
        #define REAL double
        double a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        #undef REAL
        double b = 0.0;
        """

    def c_code(self, node, name, inp, out, sub):
        _x, _y, _a = inp
        (_zout,) = out
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        if len(self.c_libraries()) <= 0:
            raise NotImplementedError()
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (2,) + gv
        else:
            return gv


_dot22scalar = Dot22Scalar()


@node_rewriter([mul])
def local_dot22_to_dot22scalar(fgraph, node):
    """
    Notes
    -----
    Previous attempts to alter this optimization to replace dot22 with
    gemm instead of dot22scalar resulted in some Scan nodes being
    duplicated and the ScanSaveMem optimization never running on them,
    resulting in highly increased memory usage. Until this issue is
    resolved, this optimization should keep using dot22scalar instead of
    gemm.

    We upcast the scalar if after the multiplication with the dot this give
    the same type.

    We execute this optimizer after the gemm optimizer. This
    allow to give more priority to gemm that give more speed up
    then this optimizer, but allow the gemm optimizer to ignore
    this op.

    TODO: support when we can reorder the mul to generate a
    dot22scalar or fix the canonizer to merge them(1 mul with multiple
    inputs)

    """
    if node.op != mul:
        return False
    i_dot22 = [x.owner and x.owner.op == _dot22 for x in node.inputs]
    if not any(i_dot22):
        return False  # no dot22
    if i_dot22.count(True) > 1:
        # TODO: try each of them.
        pass
        # return False #TODO fix
    dot22_idx = i_dot22.index(True)
    d = node.inputs[dot22_idx]
    i_scalar = [_as_scalar(x, dtype=d.dtype) for x in node.inputs]
    if not any(i_scalar):
        # Check if we can reorder the graph as this mul have a mul in inputs.
        # We support only 1 additional level of mul.
        # The canonizer should have merged those mul together.
        i_mul = [
            x.owner
            and x.owner.op == mul
            and any(_as_scalar(x_i, dtype=d.dtype) for x_i in x.owner.inputs)
            for x in node.inputs
        ]
        if not any(i_mul):
            # no scalar in input and no multiplication
            # if their was a multiplication we couls reorder the graph
            # by the associativity of the graph.
            return False

        mul_idx = i_mul.index(True)  # The first one should always work
        m = node.inputs[mul_idx]

        scalar_idx = -1
        for i, x in enumerate(m.owner.inputs):
            if _as_scalar(x, dtype=d.dtype) and (
                pytensor.scalar.upcast(x.type.dtype, d.type.dtype) == d.type.dtype
            ):
                scalar_idx = i
                break

        if scalar_idx < 0:
            _logger.info(
                f"Not optimizing dot22 with inputs {node.inputs} {[x.type for x in node.inputs]}, as the"
                " type of the scalar cannot be upcasted to the"
                " matrix type"
            )
            return False
        a = at.cast(_as_scalar(m.owner.inputs[scalar_idx], dtype=d.dtype), d.type.dtype)
        assert not a.type.ndim
        dot = _dot22scalar(d.owner.inputs[0], d.owner.inputs[1], a)

        # The other inputs to the original node that were
        # neither part of the dot22 or this mul should be
        # factors in the returned "mul" node.
        assert dot22_idx != mul_idx
        other_factors = [
            inpt for i, inpt in enumerate(node.inputs) if i not in (dot22_idx, mul_idx)
        ]
        other_m_inputs = [
            inpt for i, inpt in enumerate(m.owner.inputs) if i != scalar_idx
        ]

        return [mul(dot, *(other_factors + other_m_inputs))]

    scalar_idx = -1
    for i, x in enumerate(node.inputs):
        if (
            i != dot22_idx
            and i_scalar[i] is not None
            and (pytensor.scalar.upcast(x.type.dtype, d.type.dtype) == d.type.dtype)
        ):
            scalar_idx = i
            break
    if scalar_idx < 0:
        _logger.info(
            f"Not optimizing dot22 with inputs {node.inputs} {[x.type for x in node.inputs]}, as the type "
            "of the scalar cannot be upcasted to the matrix type"
        )
        return False
    assert scalar_idx < len(node.inputs)
    s = node.inputs[scalar_idx]
    o = copy.copy(node.inputs)
    o.remove(d)
    o.remove(s)

    a = at.cast(i_scalar[scalar_idx], d.type.dtype)
    assert not a.type.ndim
    if len(o) == 0:
        return [_dot22scalar(d.owner.inputs[0], d.owner.inputs[1], a)]
    else:
        return [mul(_dot22scalar(d.owner.inputs[0], d.owner.inputs[1], a), *o)]


# must happen after gemm as the gemm optimizer don't understant
# dot22scalar and gemm give more speed up then dot22scalar
blas_optdb.register(
    "local_dot22_to_dot22scalar",
    in2out(local_dot22_to_dot22scalar),
    "fast_run",
    position=11,
)


class BatchedDot(COp):
    """
    Computes the batched dot product of two variables:

        batched_dot(a, b)[i] = dot(a[i], b[i])
    """

    __props__ = ()

    def make_node(self, *inputs):
        inputs = list(map(at.as_tensor_variable, inputs))

        if any(not isinstance(i.type, DenseTensorType) for i in inputs):
            raise NotImplementedError("Only dense tensor types are supported")

        if len(inputs) != 2:
            raise TypeError(f"Two arguments required, but {len(inputs)} given.")
        if inputs[0].ndim not in (2, 3):
            raise TypeError(
                "Input 0 (0-indexed)"
                f" must have ndim of 2 or 3, {int(inputs[0].ndim)} given. Consider"
                " calling batched_dot instead."
            )
        if inputs[1].ndim not in (2, 3):
            raise TypeError(
                "Input 1 (0-indexed)"
                f" must have ndim of 2 or 3, {int(inputs[1].ndim)} given. Consider"
                " calling batched_dot instead."
            )

        dtype = pytensor.scalar.upcast(*[input.type.dtype for input in inputs])
        # upcast inputs to common dtype if needed
        upcasted_inputs = [at.cast(input, dtype) for input in inputs]
        out_shape = (
            (
                1
                if inputs[0].type.shape[0] == 1 or inputs[1].type.shape[0] == 1
                else None,
            )
            + inputs[0].type.shape[1:-1]
            + inputs[1].type.shape[2:]
        )
        out_shape = tuple(1 if s == 1 else None for s in out_shape)
        return Apply(self, upcasted_inputs, [tensor(dtype, shape=out_shape)])

    def perform(self, node, inp, out):
        x, y = inp
        (z,) = out

        if x.shape[0] != y.shape[0]:
            raise TypeError(
                f"Inputs [{', '.join(map(str, inp))}] must have the"
                f" same size in axis 0, but have sizes [{', '.join([str(i.shape[0]) for i in inp])}]."
            )

        shape = self.infer_shape(None, node, [i.shape for i in inp])[0]
        dtype = node.outputs[0].dtype
        z0 = z[0] = np.empty(shape, dtype=dtype)
        for i in range(z0.shape[0]):
            z0[i] = np.dot(x[i], y[i])

    def c_support_code(self, **kwargs):
        batch_gemm_defn = """
        template<typename dtype>
        bool batch_gemm(void (*gemm)(char*, char*, const int*, const int*, const int*, const dtype*, const dtype*, const int*, const dtype*, const int*, const dtype*, dtype*, const int*),
                        int type_size, PyArrayObject* xs, PyArrayObject* ys,
                        PyArrayObject* zs) {
            npy_intp *Nx = PyArray_DIMS(xs), *Sx = PyArray_STRIDES(xs);
            npy_intp *Ny = PyArray_DIMS(ys), *Sy = PyArray_STRIDES(ys);
            npy_intp *Nz = PyArray_DIMS(zs), *Sz = PyArray_STRIDES(zs);

            if (Nx[0] != Ny[0]) {
                PyErr_Format(PyExc_ValueError,
                             "Shape mismatch: batch sizes unequal."
                             " x.shape is (%d, %d, %d),"
                             " y.shape is (%d, %d, %d).",
                             Nx[0], Nx[1], Nx[2],
                             Ny[0], Ny[1], Ny[2]);
                return 1;
            }

            if (Nx[2] != Ny[1]) {
                PyErr_Format(PyExc_ValueError,
                             "Shape mismatch: summation axis sizes unequal."
                             " x.shape is (%d, %d, %d),"
                             " y.shape is (%d, %d, %d).",
                             Nx[0], Nx[1], Nx[2],
                             Ny[0], Ny[1], Ny[2]);
                return 1;
            }

            /* encode the stride structure of _x,_y,_z into a single integer. */
            int unit = 0;
            unit |= ((Sx[2] == type_size || Nx[2] == 1) ? 0x0 : (Sx[1] == type_size || Nx[1]==1) ? 0x1 : 0x2) << 8;
            unit |= ((Sy[2] == type_size || Ny[2] == 1) ? 0x0 : (Sy[1] == type_size || Ny[1]==1) ? 0x1 : 0x2) << 4;
            unit |= ((Sz[2] == type_size || Nz[2] == 1) ? 0x0 : (Sz[1] == type_size || Nz[1]==1) ? 0x1 : 0x2) << 0;

            /* create appropriate strides for malformed matrices that are row or column
             * vectors, or empty matrices.
             * In that case, the value of the stride does not really matter, but
             * some versions of BLAS insist that:
             *  - they are not smaller than the number of elements in the array,
             *  - they are not 0.
             */
            int sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : (Nx[2] + 1);
            int sx_2 = (Nx[2] > 1) ? Sx[2]/type_size : (Nx[1] + 1);
            int sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : (Ny[2] + 1);
            int sy_2 = (Ny[2] > 1) ? Sy[2]/type_size : (Ny[1] + 1);
            int sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : (Nz[2] + 1);
            int sz_2 = (Nz[2] > 1) ? Sz[2]/type_size : (Nz[1] + 1);

            dtype* x = (dtype*)PyArray_DATA(xs);
            dtype* y = (dtype*)PyArray_DATA(ys);
            dtype* z = (dtype*)PyArray_DATA(zs);

            dtype a = 1.0;
            dtype b = 0.0;
            char N = 'N';
            char T = 'T';
            int Nz1 = Nz[1], Nz2 = Nz[2], Nx2 = Nx[2];

            // loop over batch axis
            for (int i = 0; i < Nz[0]; i++) {
                switch(unit)
                {
                    case 0x000: gemm(&N, &N, &Nz2, &Nz1, &Nx2, &a, y, &sy_1, x, &sx_1, &b, z, &sz_1); break;
                    case 0x100: gemm(&N, &T, &Nz2, &Nz1, &Nx2, &a, y, &sy_1, x, &sx_2, &b, z, &sz_1); break;
                    case 0x010: gemm(&T, &N, &Nz2, &Nz1, &Nx2, &a, y, &sy_2, x, &sx_1, &b, z, &sz_1); break;
                    case 0x110: gemm(&T, &T, &Nz2, &Nz1, &Nx2, &a, y, &sy_2, x, &sx_2, &b, z, &sz_1); break;
                    case 0x001: gemm(&T, &T, &Nz1, &Nz2, &Nx2, &a, x, &sx_1, y, &sy_1, &b, z, &sz_2); break;
                    case 0x101: gemm(&N, &T, &Nz1, &Nz2, &Nx2, &a, x, &sx_2, y, &sy_1, &b, z, &sz_2); break;
                    case 0x011: gemm(&T, &N, &Nz1, &Nz2, &Nx2, &a, x, &sx_1, y, &sy_2, &b, z, &sz_2); break;
                    case 0x111: gemm(&N, &N, &Nz1, &Nz2, &Nx2, &a, x, &sx_2, y, &sy_2, &b, z, &sz_2); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); return 1;
                };
                x += Sx[0] / type_size;
                y += Sy[0] / type_size;
                z += Sz[0] / type_size;
            }

            return 0;
        }
        """
        return blas_header_text() + batch_gemm_defn

    def c_libraries(self, **kwargs):
        return ldflags()

    def c_compile_args(self, **kwargs):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return ldflags(libs=False, include_dir=True)

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        return """
        // clean up views
        Py_XDECREF(xs); xs = 0;
        Py_XDECREF(ys); ys = 0;
        Py_XDECREF(zs); zs = 0;
        """

    def c_code(self, node, name, inp, out, sub):
        _x, _y = inp
        (_z,) = out
        fail = sub["fail"]

        # generate contiguity condition
        def contiguous(var, ndim):
            strides = f"PyArray_STRIDES({var})"
            if ndim == 1:
                return f"{strides}[0] == type_size"
            return " && ".join(
                [
                    " && ".join(
                        "{strides}[{i}] > 0 && {strides}[{i}] % type_size == 0".format(
                            strides=strides, i=i
                        )
                        for i in range(1, ndim)
                    ),
                    "(%s)"
                    % " || ".join(
                        "{strides}[{i}] == type_size".format(strides=strides, i=i)
                        for i in range(1, ndim)
                    ),
                ]
            )

        x_ndim, y_ndim, z_ndim = (
            node.inputs[0].ndim,
            node.inputs[1].ndim,
            node.outputs[0].ndim,
        )

        # generate code to allocate output based on runtime input shapes
        z_dims = [f"PyArray_DIMS({_x})[0]"]
        if x_ndim == 3:
            z_dims.append(f"PyArray_DIMS({_x})[1]")
        if y_ndim == 3:
            z_dims.append(f"PyArray_DIMS({_y})[2]")
        assert len(z_dims) == z_ndim

        z_shape_correct = " && ".join(
            "PyArray_DIMS(%s)[%i] == %s" % (_z, i, dim) for i, dim in enumerate(z_dims)
        )
        z_shape = ", ".join(z_dims)
        z_contiguous = contiguous(_z, z_ndim)
        allocate = (
            """
            if (NULL == %(_z)s || !(%(z_shape_correct)s)  || !(%(z_contiguous)s))
            {
                npy_intp dims[%(z_ndim)s] = {%(z_shape)s};
                Py_XDECREF(%(_z)s);
                %(_z)s = (PyArrayObject*)PyArray_SimpleNew(
                    %(z_ndim)s, dims, PyArray_TYPE(%(_x)s));
                if(!%(_z)s) {
                    PyErr_SetString(PyExc_MemoryError,
                                    "failed to alloc BatchedDot output");
                    %(fail)s
                }
            }
        """
            % locals()
        )

        # code to reallocate inputs contiguously if necessary
        contiguate = []
        for var, ndim in [(_x, x_ndim), (_y, y_ndim)]:
            _contiguous = contiguous(var, ndim)
            contiguate.append(
                """
                if (!(%(_contiguous)s)) {
                    PyArrayObject * _copy = (PyArrayObject *) PyArray_Copy(%(var)s);
                    if (!_copy)
                        %(fail)s
                    Py_XDECREF(%(var)s);
                    %(var)s = _copy;
                }
            """
                % locals()
            )
        contiguate = "\n".join(contiguate)

        def c_dimshuffle(newname, oldname, shape):
            _fail = fail
            _shape = ", ".join(
                "1" if axis is None else "PyArray_DIMS(%s)[%i]" % (oldname, axis)
                for axis in shape
            )
            return (
                """{
                npy_intp dims[3] = {%(_shape)s};
                PyArray_Dims newshape = {dims, 3};
                %(newname)s = (PyArrayObject*)PyArray_Newshape(%(oldname)s, &newshape, NPY_ANYORDER);
                if (!%(newname)s)
                    %(_fail)s
                // make sure we didn't accidentally copy
                assert(PyArray_DATA(%(oldname)s) == PyArray_DATA(%(newname)s));
            }"""
                % locals()
            )

        # create tensor3 views for any of x, y, z that are not tensor3, so that
        # we only need to implement the tensor3-tensor3 batched dot product.
        # xs, ys and zs will point to these views, or to the original array if
        # it was already tensor3.
        # in the latter case, we artificially increase the reference count of
        # the original array so that the c_code_cleanup method can decref them
        # all indiscriminately.
        upcast = []
        if x_ndim == 3:
            upcast.append("xs = %(_x)s; Py_XINCREF(xs);")
        elif x_ndim == 2:
            upcast.append(c_dimshuffle("xs", _x, (0, None, 1)))
        if y_ndim == 3:
            upcast.append("ys = %(_y)s; Py_XINCREF(ys);")
        elif y_ndim == 2:
            upcast.append(c_dimshuffle("ys", _y, (0, 1, None)))
        if z_ndim == 3:
            upcast.append("zs = %(_z)s; Py_XINCREF(zs);")
        else:
            upcast.append(
                c_dimshuffle(
                    "zs",
                    _z,
                    (0, None if x_ndim == 2 else 1, None if y_ndim == 2 else 1),
                )
            )
        upcast = "\n".join(upcast) % locals()

        return (
            """
        int type_num = PyArray_DESCR(%(_x)s)->type_num;
        int type_size = PyArray_DESCR(%(_x)s)->elsize; // in bytes

        // xs, ys, zs will point to views onto %(_x)s, %(_y)s, %(_z)s
        PyArrayObject *xs = 0, *ys = 0, *zs = 0;

        if (PyArray_NDIM(%(_x)s) != %(x_ndim)s) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(x) != %(x_ndim)s. rank(x) is %%d.",
                         PyArray_NDIM(%(_x)s));
            %(fail)s;
        }
        if (PyArray_NDIM(%(_y)s) != %(y_ndim)s) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(y) != %(y_ndim)s. rank(y) is %%d.",
                         PyArray_NDIM(%(_y)s));
            %(fail)s;
        }
        if (%(_z)s && PyArray_NDIM(%(_z)s) != %(z_ndim)s) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(z) != %(z_ndim)s. rank(z) is %%d.",
                         PyArray_NDIM(%(_z)s));
            %(fail)s;
        }

        // allocate output
        %(allocate)s
        // reallocate any noncontiguous arrays or arrays with invalid strides
        %(contiguate)s
        // add dims to make sure everything is tensor3
        %(upcast)s
        // from here on, use xs, ys and zs as they are tensor3 and share memory
        // with the original %(_x)s, %(_y)s and %(_z)s arrays.

        if ((PyArray_DESCR(xs)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(xs)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(ys)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(ys)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(zs)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(zs)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(xs)->type_num != PyArray_DESCR(ys)->type_num)
            ||(PyArray_DESCR(xs)->type_num != PyArray_DESCR(zs)->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(x), type(y), type(z) are not all the same"); %(fail)s; }

        switch (type_num)
        {
            case NPY_FLOAT:
            if (batch_gemm<float>(sgemm_, type_size, xs, ys, zs)) {
                %(fail)s;
            }
            break;
            case NPY_DOUBLE:
            if (batch_gemm<double>(dgemm_, type_size, xs, ys, zs)) {
                %(fail)s;
            }
            break;
        }
        """
            % locals()
        )

    def c_code_cache_version(self):
        from pytensor.tensor.blas_headers import blas_header_version

        return (4, blas_header_version())

    def grad(self, inp, grads):
        x, y = inp
        (gz,) = grads
        xdim, ydim, gdim = x.type.ndim, y.type.ndim, gz.type.ndim

        # grad is a vector, so x is a matrix and y is a matrix
        if gdim == 1:
            xgrad = gz.dimshuffle(0, "x") * y
            ygrad = gz.dimshuffle(0, "x") * x

        # x is a matrix, y is a tensor3, grad is a matrix
        elif xdim == 2 and ydim == 3:
            xgrad = batched_dot(gz, y.dimshuffle(0, 2, 1))
            ygrad = x.dimshuffle(0, 1, "x") * gz.dimshuffle(0, "x", 1)

        # x is a tensor3, y is a matrix, grad is a matrix
        elif xdim == 3 and ydim == 2:
            xgrad = gz.dimshuffle(0, 1, "x") * y.dimshuffle(0, "x", 1)
            ygrad = batched_dot(x.dimshuffle(0, 2, 1), gz)

        # x is a tensor3, y is a tensor3, grad is a tensor3
        elif xdim == ydim == 3:
            xgrad = batched_dot(gz, y.dimshuffle(0, 2, 1))
            ygrad = batched_dot(x.dimshuffle(0, 2, 1), gz)

        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = specify_broadcastable(
                xgrad, *(ax for (ax, b) in enumerate(x.type.broadcastable) if b)
            )
        if ygrad.broadcastable != y.broadcastable:
            ygrad = specify_broadcastable(
                ygrad, *(ax for (ax, b) in enumerate(y.type.broadcastable) if b)
            )

        return xgrad, ygrad

    def R_op(self, inputs, eval_points):
        # R_op for batched_dot(a, b) evaluated at c for a and d for b is
        # simply batched_dot(c, b) + batched_dot(a, d)

        assert len(inputs) == 2
        assert len(eval_points) == 2
        if eval_points[0] is None and eval_points[1] is None:
            return [None]

        test_values_enabled = config.compute_test_value != "off"

        if test_values_enabled:
            try:
                iv0 = pytensor.graph.op.get_test_value(inputs[0])
            except TestValueError:
                pytensor.graph.op.missing_test_message(
                    "first input passed to BatchedDot.R_op has no test value"
                )
                test_values_enabled = False

            try:
                iv1 = pytensor.graph.op.get_test_value(inputs[1])
            except TestValueError:
                pytensor.graph.op.missing_test_message(
                    "second input passed to BatchedDot.R_op has no test value"
                )
                test_values_enabled = False

            if eval_points[0]:
                try:
                    ev0 = pytensor.graph.op.get_test_value(eval_points[0])
                except TestValueError:
                    pytensor.graph.op.missing_test_message(
                        "first eval point passed to BatchedDot.R_op "
                        "has no test value"
                    )
                    test_values_enabled = False
            if eval_points[1]:
                try:
                    ev1 = pytensor.graph.op.get_test_value(eval_points[1])
                except TestValueError:
                    pytensor.graph.op.missing_test_message(
                        "second eval point passed to BatchedDot.R_op "
                        "has no test value"
                    )
                    test_values_enabled = False

        if test_values_enabled:
            input_values = [iv0, iv1]
            eval_point_values = [ev0, ev1]

            for i in range(2):
                if (
                    eval_point_values[i] is not None
                    and input_values[i].shape != eval_point_values[i].shape
                ):
                    raise ValueError(
                        "input "
                        + str(i)
                        + " and eval_point "
                        + str(i)
                        + " to BatchedDot.R_op should have the same shape, but "
                        f"their shapes are {input_values[i].shape} and {eval_point_values[i].shape}, respectively"
                    )
        if eval_points[0]:
            t1 = self(eval_points[0], inputs[1])
        if eval_points[1]:
            t2 = self(inputs[0], eval_points[1])

        if eval_points[0] and eval_points[1]:
            return [t1 + t2]
        elif eval_points[0]:
            return [t1]
        else:
            return [t2]

    def infer_shape(self, fgraph, node, shapes):
        for shape_ in shapes:
            if len(shape_) not in (2, 3):
                raise NotImplementedError()
        xshp, yshp = shapes
        return [xshp[:-1] + yshp[2:]]


_batched_dot = BatchedDot()


# from opt import register_specialize, register_canonicalize
# @register_specialize
@node_rewriter([sub, add])
def local_print_as_we_go_along(fgraph, node):
    if node.op in (sub, add):
        debugprint(node)


def batched_dot(a, b):
    """Compute the batched dot product of two variables.

    I.e.:

        batched_dot(a, b)[i] = dot(a[i], b[i])

    Note that this batched_dot function does one of three things, in the
    following sequence:

        1.  If either a or b is a vector, it returns the batched elementwise
            product without calling the PyTensor BatchedDot op.

        2.  If both a and b have either 2 or 3 dimensions, it calls PyTensor's
            BatchedDot op on a and b.

        3.  If either a or b has more than 3 dimensions, it calls PyTensor's
            batched_tensordot function with appropriate axes. The
            batched_tensordot function expresses high-dimensional batched
            dot products in terms of batched matrix-matrix dot products, so
            it may be possible to further optimize for performance.
    """
    a, b = at.as_tensor_variable(a), at.as_tensor_variable(b)

    if a.ndim == 0:
        raise TypeError("a must have at least one (batch) axis")
    elif b.ndim == 0:
        raise TypeError("b must have at least one (batch) axis")
    elif a.ndim == 1:
        return a.dimshuffle(*([0] + ["x"] * (b.ndim - 1))) * b
    elif b.ndim == 1:
        return a * b.dimshuffle(*([0] + ["x"] * (a.ndim - 1)))
    elif a.ndim > 3 or b.ndim > 3:
        return batched_tensordot(a, b, [[a.ndim - 1], [np.maximum(1, b.ndim - 2)]])
    else:
        # avoid circular import
        return _batched_dot(a, b)


def batched_tensordot(x, y, axes=2):
    """Compute a batched tensordot product.

    A hybrid of batched_dot and tensordot, this function computes the
    tensordot product between the two tensors, by iterating over the
    first dimension to perform a sequence of tensordots.

    Parameters
    ----------
    x: TensorVariable
        A tensor with sizes e.g.: for 3D (dim1, dim3, dim2)
    y: TensorVariable
        A tensor with sizes e.g.: for 3D (dim1, dim2, dim4)
    axes: int or array-like of length 2
        If an integer, the number of axes to sum over.
        If an array, it must have two array elements containing the axes to sum
        over in each tensor.

        If an integer i, it is converted to an array containing
        the last i dimensions of the first tensor and the first
        i dimensions of the second tensor (excluding the first
        (batch) dimension):
            axes = [list(range(a.ndim - i, b.ndim)), list(range(1,i+1))]

        If an array, its two elements must contain compatible axes
        of the two tensors. For example, [[1, 2], [2, 4]] means sum
        over the 2nd and 3rd axes of a and the 3rd and 5th axes of b.
        (Remember axes are zero-indexed!) The 2nd axis of a and the
        3rd axis of b must have the same shape; the same is true for
        the 3rd axis of a and the 5th axis of b.

    Like tensordot, this function uses a series of dimshuffles and
    reshapes to reduce the tensor dot product to a matrix or vector
    dot product.  Finally, it calls batched_dot to compute the result.
    """
    from pytensor.tensor.math import _tensordot_as_dot

    return _tensordot_as_dot(x, y, axes, dot=batched_dot, batched=True)
