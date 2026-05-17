"""
python/_core.py
---------------
Loads the compiled islpstat.dll and declares the ctypes signatures for every
exported C function.  Nothing statistical lives here — this is pure plumbing.

On Windows the shared library is islpstat.dll, produced by build.bat.
"""

import ctypes
import os
import numpy as np

# ------------------------------------------------------------------
# Load the DLL
# ------------------------------------------------------------------
_here     = os.path.dirname(os.path.abspath(__file__))
_dll_path = os.path.join(_here, '..', 'islpstat.dll')

try:
    _lib = ctypes.CDLL(_dll_path)
except OSError as e:
    raise ImportError(
        f"Could not load islpstat.dll from {_dll_path}.\n"
        "Run build.bat first to compile the C library.\n"
        f"Original error: {e}"
    )

_dbl_p = ctypes.POINTER(ctypes.c_double)
_int_p = ctypes.POINTER(ctypes.c_int)
_dbl   = ctypes.c_double
_int   = ctypes.c_int

# ------------------------------------------------------------------
# C struct mirrors
# ------------------------------------------------------------------

class _LinRegResult(ctypes.Structure):
    _fields_ = [
        ("n",         _int),
        ("p",         _int),
        ("beta",      _dbl_p),
        ("se",        _dbl_p),
        ("t_stat",    _dbl_p),
        ("p_value",   _dbl_p),
        ("ci_lower",  _dbl_p),
        ("ci_upper",  _dbl_p),
        ("rse",       _dbl),
        ("r_squared", _dbl),
        ("adj_r2",    _dbl),
        ("f_stat",    _dbl),
        ("f_pvalue",  _dbl),
        ("XtXinv",    _dbl_p),
    ]

class _LogRegResult(ctypes.Structure):
    _fields_ = [
        ("n",              _int),
        ("p",              _int),
        ("beta",           _dbl_p),
        ("se",             _dbl_p),
        ("z_stat",         _dbl_p),
        ("p_value",        _dbl_p),
        ("null_deviance",  _dbl),
        ("deviance",       _dbl),
        ("aic",            _dbl),
        ("converged",      _int),
        ("n_iter",         _int),
    ]

# LDA / QDA / NaiveBayes structs (opaque — we only pass pointers)
class _LDAModel(ctypes.Structure):
    _fields_ = [
        ("n",            _int),
        ("p",            _int),
        ("K",            _int),
        ("class_labels", _int_p),
        ("pi",           _dbl_p),
        ("mu",           _dbl_p),
        ("Sigma_inv",    _dbl_p),
        ("lda_const",    _dbl_p),
    ]

class _QDAModel(ctypes.Structure):
    _fields_ = [
        ("n",            _int),
        ("p",            _int),
        ("K",            _int),
        ("class_labels", _int_p),
        ("pi",           _dbl_p),
        ("mu",           _dbl_p),
        ("Sigma_inv",    _dbl_p),
        ("log_det",      _dbl_p),
    ]

class _NaiveBayesModel(ctypes.Structure):
    _fields_ = [
        ("n",            _int),
        ("p",            _int),
        ("K",            _int),
        ("class_labels", _int_p),
        ("pi",           _dbl_p),
        ("mu",           _dbl_p),
        ("var",          _dbl_p),
    ]

class _BinaryConfusion(ctypes.Structure):
    _fields_ = [
        ("TP",          _int),
        ("TN",          _int),
        ("FP",          _int),
        ("FN",          _int),
        ("n",           _int),
        ("sensitivity", _dbl),
        ("specificity", _dbl),
        ("error_rate",  _dbl),
        ("precision",   _dbl),
    ]

# ------------------------------------------------------------------
# linreg
# ------------------------------------------------------------------
_lib.linreg_fit.restype  = ctypes.POINTER(_LinRegResult)
_lib.linreg_fit.argtypes = [_dbl_p, _dbl_p, _int, _int]

_lib.linreg_free.restype  = None
_lib.linreg_free.argtypes = [ctypes.POINTER(_LinRegResult)]

_lib.linreg_predict.restype  = None
_lib.linreg_predict.argtypes = [ctypes.POINTER(_LinRegResult), _dbl_p, _dbl_p, _int]

_lib.linreg_predict_ci.restype  = None
_lib.linreg_predict_ci.argtypes = [
    ctypes.POINTER(_LinRegResult), _dbl_p, _int, _dbl, _dbl_p, _dbl_p]

_lib.linreg_predict_pi.restype  = None
_lib.linreg_predict_pi.argtypes = [
    ctypes.POINTER(_LinRegResult), _dbl_p, _int, _dbl, _dbl_p, _dbl_p]

# ------------------------------------------------------------------
# logreg
# ------------------------------------------------------------------
_lib.logreg_fit.restype  = ctypes.POINTER(_LogRegResult)
_lib.logreg_fit.argtypes = [_dbl_p, _dbl_p, _int, _int, _int, _dbl]

_lib.logreg_free.restype  = None
_lib.logreg_free.argtypes = [ctypes.POINTER(_LogRegResult)]

_lib.logreg_predict_prob.restype  = None
_lib.logreg_predict_prob.argtypes = [
    ctypes.POINTER(_LogRegResult), _dbl_p, _dbl_p, _int]

_lib.logreg_predict.restype  = None
_lib.logreg_predict.argtypes = [
    ctypes.POINTER(_LogRegResult), _dbl_p, _int_p, _int, _dbl]

# ------------------------------------------------------------------
# LDA
# ------------------------------------------------------------------
_lib.lda_fit.restype  = ctypes.POINTER(_LDAModel)
_lib.lda_fit.argtypes = [_dbl_p, _int_p, _int, _int, _int]

_lib.lda_free.restype  = None
_lib.lda_free.argtypes = [ctypes.POINTER(_LDAModel)]

_lib.lda_predict.restype  = None
_lib.lda_predict.argtypes = [ctypes.POINTER(_LDAModel), _dbl_p, _int_p, _int]

_lib.lda_posterior.restype  = None
_lib.lda_posterior.argtypes = [ctypes.POINTER(_LDAModel), _dbl_p, _dbl_p, _int]

# ------------------------------------------------------------------
# QDA
# ------------------------------------------------------------------
_lib.qda_fit.restype  = ctypes.POINTER(_QDAModel)
_lib.qda_fit.argtypes = [_dbl_p, _int_p, _int, _int, _int]

_lib.qda_free.restype  = None
_lib.qda_free.argtypes = [ctypes.POINTER(_QDAModel)]

_lib.qda_predict.restype  = None
_lib.qda_predict.argtypes = [ctypes.POINTER(_QDAModel), _dbl_p, _int_p, _int]

_lib.qda_posterior.restype  = None
_lib.qda_posterior.argtypes = [ctypes.POINTER(_QDAModel), _dbl_p, _dbl_p, _int]

# ------------------------------------------------------------------
# Naive Bayes
# ------------------------------------------------------------------
_lib.nb_fit.restype  = ctypes.POINTER(_NaiveBayesModel)
_lib.nb_fit.argtypes = [_dbl_p, _int_p, _int, _int, _int]

_lib.nb_free.restype  = None
_lib.nb_free.argtypes = [ctypes.POINTER(_NaiveBayesModel)]

_lib.nb_predict.restype  = None
_lib.nb_predict.argtypes = [ctypes.POINTER(_NaiveBayesModel), _dbl_p, _int_p, _int]

_lib.nb_posterior.restype  = None
_lib.nb_posterior.argtypes = [ctypes.POINTER(_NaiveBayesModel), _dbl_p, _dbl_p, _int]

# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------
_lib.confusion_matrix_binary.restype  = _BinaryConfusion
_lib.confusion_matrix_binary.argtypes = [_int_p, _int_p, _int, _int]

_lib.accuracy.restype  = _dbl
_lib.accuracy.argtypes = [_int_p, _int_p, _int]

# ------------------------------------------------------------------
# Helpers: numpy array -> ctypes pointer (no copy)
# ------------------------------------------------------------------
def _dbl_ptr(arr):
    """Return a ctypes double* pointing at a contiguous float64 numpy array."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return a.ctypes.data_as(_dbl_p), a   # return `a` to keep alive

def _int_ptr(arr):
    """Return a ctypes int* pointing at a contiguous int32 numpy array."""
    a = np.ascontiguousarray(arr, dtype=np.int32)
    return a.ctypes.data_as(_int_p), a

def _out_dbl(n):
    """Allocate an (n,) float64 output buffer; return (ptr, array)."""
    a = np.zeros(n, dtype=np.float64)
    return a.ctypes.data_as(_dbl_p), a

def _out_int(n):
    """Allocate an (n,) int32 output buffer; return (ptr, array)."""
    a = np.zeros(n, dtype=np.int32)
    return a.ctypes.data_as(_int_p), a
