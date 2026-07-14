"""
python/_core.py  —  ctypes bindings for islpstat.dll' if os.name=='nt' else os.path.join(_here,'..','islpstat.so
"""
import ctypes, os, numpy as np

_here     = os.path.dirname(os.path.abspath(__file__))
_dll_path = os.path.join(_here, '..', 'islpstat.dll') if os.name == 'nt' else os.path.join(_here, '..', 'islpstat.so')
try:
    _lib = ctypes.CDLL(_dll_path)
except OSError as e:
    raise ImportError(f"Could not load islpstat.dll' if os.name=='nt' else os.path.join(_here,'..','islpstat.so from {_dll_path}.\n"
                      "Run build.bat first.\nOriginal error: {e}")

_dbl_p = ctypes.POINTER(ctypes.c_double)
_int_p = ctypes.POINTER(ctypes.c_int)
_dbl   = ctypes.c_double
_int   = ctypes.c_int

# ── structs ──────────────────────────────────────────────────────────

class _LinRegResult(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("beta",_dbl_p),("se",_dbl_p),
                ("t_stat",_dbl_p),("p_value",_dbl_p),("ci_lower",_dbl_p),
                ("ci_upper",_dbl_p),("rse",_dbl),("r_squared",_dbl),
                ("adj_r2",_dbl),("f_stat",_dbl),("f_pvalue",_dbl),("XtXinv",_dbl_p)]

class _LogRegResult(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("beta",_dbl_p),("se",_dbl_p),
                ("z_stat",_dbl_p),("p_value",_dbl_p),("null_deviance",_dbl),
                ("deviance",_dbl),("aic",_dbl),("converged",_int),("n_iter",_int)]

class _LDAModel(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("K",_int),("class_labels",_int_p),
                ("pi",_dbl_p),("mu",_dbl_p),("Sigma_inv",_dbl_p),("lda_const",_dbl_p)]

class _QDAModel(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("K",_int),("class_labels",_int_p),
                ("pi",_dbl_p),("mu",_dbl_p),("Sigma_inv",_dbl_p),("log_det",_dbl_p)]

class _NaiveBayesModel(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("K",_int),("class_labels",_int_p),
                ("pi",_dbl_p),("mu",_dbl_p),("var",_dbl_p)]

class _BinaryConfusion(ctypes.Structure):
    _fields_ = [("TP",_int),("TN",_int),("FP",_int),("FN",_int),("n",_int),
                ("sensitivity",_dbl),("specificity",_dbl),("error_rate",_dbl),("precision",_dbl)]

class _BinaryMetrics(ctypes.Structure):
    _fields_ = [("TP",_int),("TN",_int),("FP",_int),("FN",_int),("n",_int),
                ("sensitivity",_dbl),("specificity",_dbl),("precision",_dbl),
                ("npv",_dbl),("fdr",_dbl),("fpr",_dbl),("f1",_dbl),
                ("balanced_accuracy",_dbl),("mcc",_dbl),("error_rate",_dbl)]

class _LinRegDiag(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("residuals",_dbl_p),("leverage",_dbl_p),
                ("std_residuals",_dbl_p),("ext_studentized",_dbl_p),
                ("cooks_d",_dbl_p),("dffits",_dbl_p),("avg_leverage",_dbl),
                ("outlier_flag",_int_p),("high_leverage_flag",_int_p),
                ("influential_flag",_int_p)]

class _ROCCurve(ctypes.Structure):
    _fields_ = [("fpr",_dbl_p),("tpr",_dbl_p),("thresholds",_dbl_p),
                ("n_points",_int),("auc",_dbl)]

class _KNNClassifier(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("K",_int),("n_classes",_int),
                ("class_labels",_int_p),("X_train",_dbl_p),("y_train",_int_p)]

class _KNNRegressor(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("K",_int),("X_train",_dbl_p),("y_train",_dbl_p)]

class _PoissonResult(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("beta",_dbl_p),("se",_dbl_p),
                ("z_stat",_dbl_p),("p_value",_dbl_p),("null_deviance",_dbl),
                ("deviance",_dbl),("aic",_dbl),("converged",_int),("n_iter",_int)]

class _MultiLogRegResult(ctypes.Structure):
    _fields_ = [("n",_int),("p",_int),("K",_int),("class_labels",_int_p),
                ("beta_mat",_dbl_p),("se_mat",_dbl_p),("z_mat",_dbl_p),
                ("p_mat",_dbl_p),("null_deviance",_dbl),("deviance",_dbl),
                ("aic",_dbl),("converged",_int),("n_iter",_int)]

# ── linreg ────────────────────────────────────────────────────────────
_lib.linreg_fit.restype  = ctypes.POINTER(_LinRegResult)
_lib.linreg_fit.argtypes = [_dbl_p,_dbl_p,_int,_int]
_lib.linreg_free.restype  = None; _lib.linreg_free.argtypes = [ctypes.POINTER(_LinRegResult)]
_lib.linreg_predict.restype=None; _lib.linreg_predict.argtypes=[ctypes.POINTER(_LinRegResult),_dbl_p,_dbl_p,_int]
_lib.linreg_predict_ci.restype=None; _lib.linreg_predict_ci.argtypes=[ctypes.POINTER(_LinRegResult),_dbl_p,_int,_dbl,_dbl_p,_dbl_p]
_lib.linreg_predict_pi.restype=None; _lib.linreg_predict_pi.argtypes=[ctypes.POINTER(_LinRegResult),_dbl_p,_int,_dbl,_dbl_p,_dbl_p]

# ── linreg_diag ───────────────────────────────────────────────────────
_lib.linreg_diagnostics.restype  = ctypes.POINTER(_LinRegDiag)
_lib.linreg_diagnostics.argtypes = [ctypes.POINTER(_LinRegResult),_dbl_p,_dbl_p,_int]
_lib.linreg_diag_free.restype=None; _lib.linreg_diag_free.argtypes=[ctypes.POINTER(_LinRegDiag)]

# ── logreg ────────────────────────────────────────────────────────────
_lib.logreg_fit.restype  = ctypes.POINTER(_LogRegResult)
_lib.logreg_fit.argtypes = [_dbl_p,_dbl_p,_int,_int,_int,_dbl]
_lib.logreg_free.restype=None; _lib.logreg_free.argtypes=[ctypes.POINTER(_LogRegResult)]
_lib.logreg_predict_prob.restype=None; _lib.logreg_predict_prob.argtypes=[ctypes.POINTER(_LogRegResult),_dbl_p,_dbl_p,_int]
_lib.logreg_predict.restype=None; _lib.logreg_predict.argtypes=[ctypes.POINTER(_LogRegResult),_dbl_p,_int_p,_int,_dbl]

# ── LDA / QDA / NaiveBayes ────────────────────────────────────────────
_lib.lda_fit.restype=ctypes.POINTER(_LDAModel); _lib.lda_fit.argtypes=[_dbl_p,_int_p,_int,_int,_int]
_lib.lda_free.restype=None; _lib.lda_free.argtypes=[ctypes.POINTER(_LDAModel)]
_lib.lda_predict.restype=None; _lib.lda_predict.argtypes=[ctypes.POINTER(_LDAModel),_dbl_p,_int_p,_int]
_lib.lda_posterior.restype=None; _lib.lda_posterior.argtypes=[ctypes.POINTER(_LDAModel),_dbl_p,_dbl_p,_int]
_lib.qda_fit.restype=ctypes.POINTER(_QDAModel); _lib.qda_fit.argtypes=[_dbl_p,_int_p,_int,_int,_int]
_lib.qda_free.restype=None; _lib.qda_free.argtypes=[ctypes.POINTER(_QDAModel)]
_lib.qda_predict.restype=None; _lib.qda_predict.argtypes=[ctypes.POINTER(_QDAModel),_dbl_p,_int_p,_int]
_lib.qda_posterior.restype=None; _lib.qda_posterior.argtypes=[ctypes.POINTER(_QDAModel),_dbl_p,_dbl_p,_int]
_lib.nb_fit.restype=ctypes.POINTER(_NaiveBayesModel); _lib.nb_fit.argtypes=[_dbl_p,_int_p,_int,_int,_int]
_lib.nb_free.restype=None; _lib.nb_free.argtypes=[ctypes.POINTER(_NaiveBayesModel)]
_lib.nb_predict.restype=None; _lib.nb_predict.argtypes=[ctypes.POINTER(_NaiveBayesModel),_dbl_p,_int_p,_int]
_lib.nb_posterior.restype=None; _lib.nb_posterior.argtypes=[ctypes.POINTER(_NaiveBayesModel),_dbl_p,_dbl_p,_int]

# ── classify_ext ─────────────────────────────────────────────────────
_lib.binary_metrics.restype  = _BinaryMetrics
_lib.binary_metrics.argtypes = [_int_p,_int_p,_int,_int]
_lib.roc_curve.restype  = ctypes.POINTER(_ROCCurve)
_lib.roc_curve.argtypes = [_int_p,_dbl_p,_int,_int]
_lib.roc_free.restype=None; _lib.roc_free.argtypes=[ctypes.POINTER(_ROCCurve)]
_lib.auc_score.restype=_dbl; _lib.auc_score.argtypes=[_int_p,_dbl_p,_int,_int]
_lib.accuracy_score.restype=_dbl; _lib.accuracy_score.argtypes=[_int_p,_int_p,_int]
_lib.confusion_matrix_binary.restype=_BinaryConfusion; _lib.confusion_matrix_binary.argtypes=[_int_p,_int_p,_int,_int]
_lib.accuracy.restype=_dbl; _lib.accuracy.argtypes=[_int_p,_int_p,_int]

# ── KNN ───────────────────────────────────────────────────────────────
_lib.knn_classifier_fit.restype=ctypes.POINTER(_KNNClassifier); _lib.knn_classifier_fit.argtypes=[_dbl_p,_int_p,_int,_int,_int]
_lib.knn_classifier_free.restype=None; _lib.knn_classifier_free.argtypes=[ctypes.POINTER(_KNNClassifier)]
_lib.knn_classifier_predict.restype=None; _lib.knn_classifier_predict.argtypes=[ctypes.POINTER(_KNNClassifier),_dbl_p,_int_p,_int]
_lib.knn_classifier_proba.restype=None; _lib.knn_classifier_proba.argtypes=[ctypes.POINTER(_KNNClassifier),_dbl_p,_dbl_p,_int]
_lib.knn_regressor_fit.restype=ctypes.POINTER(_KNNRegressor); _lib.knn_regressor_fit.argtypes=[_dbl_p,_dbl_p,_int,_int,_int]
_lib.knn_regressor_free.restype=None; _lib.knn_regressor_free.argtypes=[ctypes.POINTER(_KNNRegressor)]
_lib.knn_regressor_predict.restype=None; _lib.knn_regressor_predict.argtypes=[ctypes.POINTER(_KNNRegressor),_dbl_p,_dbl_p,_int]

# ── Poisson ───────────────────────────────────────────────────────────
_lib.poisson_fit.restype=ctypes.POINTER(_PoissonResult); _lib.poisson_fit.argtypes=[_dbl_p,_dbl_p,_int,_int,_int,_dbl]
_lib.poisson_free.restype=None; _lib.poisson_free.argtypes=[ctypes.POINTER(_PoissonResult)]
_lib.poisson_predict.restype=None; _lib.poisson_predict.argtypes=[ctypes.POINTER(_PoissonResult),_dbl_p,_dbl_p,_int]

# ── Multinomial Logistic ──────────────────────────────────────────────
_lib.multilogreg_fit.restype=ctypes.POINTER(_MultiLogRegResult); _lib.multilogreg_fit.argtypes=[_dbl_p,_int_p,_int,_int,_int,_int,_dbl]
_lib.multilogreg_free.restype=None; _lib.multilogreg_free.argtypes=[ctypes.POINTER(_MultiLogRegResult)]
_lib.multilogreg_predict_proba.restype=None; _lib.multilogreg_predict_proba.argtypes=[ctypes.POINTER(_MultiLogRegResult),_dbl_p,_dbl_p,_int]
_lib.multilogreg_predict.restype=None; _lib.multilogreg_predict.argtypes=[ctypes.POINTER(_MultiLogRegResult),_dbl_p,_int_p,_int]

# ── helpers ───────────────────────────────────────────────────────────
def _dbl_ptr(arr):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return a.ctypes.data_as(_dbl_p), a

def _int_ptr(arr):
    a = np.ascontiguousarray(arr, dtype=np.int32)
    return a.ctypes.data_as(_int_p), a

def _out_dbl(n):
    a = np.zeros(n, dtype=np.float64)
    return a.ctypes.data_as(_dbl_p), a

def _out_int(n):
    a = np.zeros(n, dtype=np.int32)
    return a.ctypes.data_as(_int_p), a
