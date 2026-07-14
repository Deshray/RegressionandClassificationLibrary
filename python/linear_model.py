"""
python/linear_model.py  —  Full ISLP Chapter 3 toolkit.
"""
import numpy as np, ctypes
from ._core import (_lib,_LinRegResult,_LinRegDiag,_dbl_ptr,_int_ptr,_out_dbl,_out_int)

def _stars(p):
    if p<0.001: return '***'
    elif p<0.01: return '**'
    elif p<0.05: return '*'
    elif p<0.1: return '.'
    return ' '

def _f_pvalue_py(F_stat, df1, df2):
    _lib.f_pvalue.restype=ctypes.c_double
    _lib.f_pvalue.argtypes=[ctypes.c_double,ctypes.c_int,ctypes.c_int]
    return float(_lib.f_pvalue(ctypes.c_double(max(0.0,F_stat)),ctypes.c_int(max(1,df1)),ctypes.c_int(max(1,df2))))

class LinearRegression:
    """OLS linear regression — full ISLP Chapter 3 methodology."""
    def __init__(self,feature_names=None):
        self.feature_names=feature_names; self._result_ptr=None
        self._X_train=None; self._y_train=None

    def fit(self,X,y,feature_names=None):
        X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
        if X.ndim==1: X=X.reshape(-1,1)
        n,p=X.shape
        if feature_names: self.feature_names=feature_names
        if not self.feature_names: self.feature_names=[f'X{j+1}' for j in range(p)]
        self._X_train=X.copy(); self._y_train=y.copy(); self._n=n; self._p=p
        if self._result_ptr is not None: _lib.linreg_free(self._result_ptr)
        Xp,_X=_dbl_ptr(X.ravel()); yp,_y=_dbl_ptr(y)
        self._result_ptr=_lib.linreg_fit(Xp,yp,ctypes.c_int(n),ctypes.c_int(p))
        if not self._result_ptr: raise RuntimeError("linreg_fit failed")
        r=self._result_ptr.contents; q=p+1
        self.intercept_=r.beta[0]
        self.coef_=np.array([r.beta[j] for j in range(1,q)])
        self.se_=np.array([r.se[j] for j in range(q)])
        self.t_stat_=np.array([r.t_stat[j] for j in range(q)])
        self.p_value_=np.array([r.p_value[j] for j in range(q)])
        self.ci_lower_=np.array([r.ci_lower[j] for j in range(q)])
        self.ci_upper_=np.array([r.ci_upper[j] for j in range(q)])
        self.rse_=r.rse; self.r_squared_=r.r_squared; self.adj_r2_=r.adj_r2
        self.f_stat_=r.f_stat; self.f_pvalue_=r.f_pvalue
        return self

    def predict(self,X):
        X=self._vX(X); m=X.shape[0]; op,out=_out_dbl(m); Xp,_X=_dbl_ptr(X.ravel())
        _lib.linreg_predict(self._result_ptr,Xp,op,ctypes.c_int(m)); return out

    def predict_ci(self,X,alpha=0.05):
        X=self._vX(X); m=X.shape[0]; lp,lo=_out_dbl(m); hp,hi=_out_dbl(m); Xp,_X=_dbl_ptr(X.ravel())
        _lib.linreg_predict_ci(self._result_ptr,Xp,ctypes.c_int(m),ctypes.c_double(alpha),lp,hp); return lo,hi

    def predict_pi(self,X,alpha=0.05):
        X=self._vX(X); m=X.shape[0]; lp,lo=_out_dbl(m); hp,hi=_out_dbl(m); Xp,_X=_dbl_ptr(X.ravel())
        _lib.linreg_predict_pi(self._result_ptr,Xp,ctypes.c_int(m),ctypes.c_double(alpha),lp,hp); return lo,hi

    @property
    def aic(self):
        rss=self.rse_**2*(self._n-self._p-1)
        return self._n*np.log(rss/self._n)+2.0*(self._p+2)
    @property
    def bic(self):
        rss=self.rse_**2*(self._n-self._p-1)
        return self._n*np.log(rss/self._n)+np.log(self._n)*(self._p+2)
    def mallows_cp(self,sigma2_full):
        rss=self.rse_**2*(self._n-self._p-1)
        return rss/sigma2_full-self._n+2.0*(self._p+1)

    def partial_ftest(self,X_reduced,y=None):
        """Partial F-test for nested models (ISLP §3.2.2 Eq 3.24)."""
        if y is None: y=self._y_train
        X_red=np.asarray(X_reduced,dtype=np.float64)
        if X_red.ndim==1: X_red=X_red.reshape(-1,1)
        m_red=LinearRegression().fit(X_red,y)
        p_full=self._p; p_red=X_red.shape[1]; q=p_full-p_red; df2=self._n-p_full-1
        rss_full=self.rse_**2*df2; rss_red=m_red.rse_**2*(self._n-p_red-1)
        F=((rss_red-rss_full)/q)/(rss_full/df2) if q>0 and df2>0 else 0.0
        return dict(F=round(F,6),p_value=round(_f_pvalue_py(F,q,df2),8),
                    df_numerator=q,df_denominator=df2,rss_full=rss_full,rss_reduced=rss_red)

    def vif(self,X=None):
        """VIF for each predictor (ISLP §3.3.3). VIF>5 concerning, >10 severe."""
        if X is None: X=self._X_train
        X=np.asarray(X,dtype=np.float64)
        if X.ndim==1: X=X.reshape(-1,1)
        p=X.shape[1]; vifs={}
        for j in range(p):
            others=[k for k in range(p) if k!=j]
            if not others: vifs[self.feature_names[j]]=float('nan'); continue
            m_j=LinearRegression().fit(X[:,others],X[:,j])
            r2=m_j.r_squared_
            vifs[self.feature_names[j]]=round(1.0/(1.0-r2) if r2<1-1e-12 else float('inf'),4)
        return vifs

    def diagnostics(self,X=None,y=None):
        """All §3.3.3 diagnostics: leverage, studentized residuals, Cook's D, DFFITS."""
        if X is None: X=self._X_train
        if y is None: y=self._y_train
        X=np.ascontiguousarray(X,dtype=np.float64)
        if X.ndim==1: X=X.reshape(-1,1)
        n=X.shape[0]; Xp,_X=_dbl_ptr(X.ravel()); yp,_y=_dbl_ptr(y)
        dptr=_lib.linreg_diagnostics(self._result_ptr,Xp,yp,ctypes.c_int(n))
        if not dptr: raise RuntimeError("linreg_diagnostics failed")
        d=dptr.contents
        result=dict(
            residuals         =np.array([d.residuals[i]       for i in range(n)]),
            leverage          =np.array([d.leverage[i]        for i in range(n)]),
            std_residuals     =np.array([d.std_residuals[i]   for i in range(n)]),
            ext_studentized   =np.array([d.ext_studentized[i] for i in range(n)]),
            cooks_d           =np.array([d.cooks_d[i]         for i in range(n)]),
            dffits            =np.array([d.dffits[i]          for i in range(n)]),
            avg_leverage      =d.avg_leverage,
            outliers          =np.where([d.outlier_flag[i]       for i in range(n)])[0].tolist(),
            high_leverage     =np.where([d.high_leverage_flag[i] for i in range(n)])[0].tolist(),
            influential       =np.where([d.influential_flag[i]   for i in range(n)])[0].tolist(),
        )
        _lib.linreg_diag_free(dptr); return result

    def diagnostics_summary(self,X=None,y=None):
        diag=self.diagnostics(X,y)
        SEP="="*62
        print(SEP); print("  Regression Diagnostics  —  ISLP §3.3.3"); print(SEP)
        print(f"  n={self._n}  p={self._p}  avg_leverage={(self._p+1)/self._n:.4f}")
        print()
        print(f"  Outliers (|t_i|>3):           {diag['outliers'] or 'none'}")
        print(f"  High leverage (h>2(p+1)/n):   {diag['high_leverage'] or 'none'}")
        print(f"  Influential (Cook's D>1):      {diag['influential'] or 'none'}")
        print(); print(f"  {'i':>5}  {'Residual':>10}  {'Leverage':>10}  {'Std.Res':>10}  {'Cook D':>10}")
        print("  "+"-"*54)
        for i in range(min(15,self._n)):
            print(f"  {i:>5}  {diag['residuals'][i]:>10.4f}  {diag['leverage'][i]:>10.4f}  "
                  f"{diag['std_residuals'][i]:>10.4f}  {diag['cooks_d'][i]:>10.4f}")
        if self._n>15: print(f"  ... ({self._n-15} more)")
        print(SEP)

    def residual_plot_data(self,X=None,y=None):
        if X is None: X=self._X_train
        if y is None: y=self._y_train
        fitted=self.predict(X); return fitted, y-fitted

    def summary(self):
        if self._result_ptr is None: print("Not fitted."); return
        SEP="="*74; LINE="-"*74
        print(SEP); print("  OLS Linear Regression  —  ISLP Chapter 3"); print(SEP)
        print(f"  n={self._n}  p={self._p}  df_resid={self._n-self._p-1}")
        print()
        names=['(Intercept)']+list(self.feature_names)
        betas=[self.intercept_]+list(self.coef_)
        print(f"  {'':20} {'Estimate':>12} {'Std.Error':>12} {'t value':>10} {'Pr(>|t|)':>12}   {'95% CI':>24}")
        print("  "+LINE)
        for j,nm in enumerate(names):
            pv=self.p_value_[j]
            ci=f"[{self.ci_lower_[j]:.4f}, {self.ci_upper_[j]:.4f}]"
            print(f"  {nm:20} {betas[j]:>12.4f} {self.se_[j]:>12.4f} {self.t_stat_[j]:>10.3f} {pv:>12.4e}  {_stars(pv):3}  {ci:>24}")
        print()
        print("  Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")
        print(); print("  "+LINE)
        print(f"  RSE={self.rse_:.4f} (df={self._n-self._p-1})")
        print(f"  R²={self.r_squared_:.4f}  Adj R²={self.adj_r2_:.4f}")
        print(f"  F={self.f_stat_:.4f} on {self._p} and {self._n-self._p-1} DF, p={self.f_pvalue_:.4e}")
        print(f"  AIC={self.aic:.2f}  BIC={self.bic:.2f}")
        print(SEP)

    def _vX(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim==1: X=X.reshape(-1,1)
        assert X.shape[1]==self._p; return np.ascontiguousarray(X)

    def __del__(self):
        if self._result_ptr is not None:
            try: _lib.linreg_free(self._result_ptr)
            except: pass


# =====================================================================
# Variable Selection  (ISLP §3.2.2)
# =====================================================================

def forward_selection(X,y,criterion='aic',feature_names=None,verbose=False):
    """Forward selection: add one predictor at a time, pick best by criterion."""
    X=np.asarray(X,dtype=np.float64)
    if X.ndim==1: X=X.reshape(-1,1)
    n,p=X.shape
    if feature_names is None: feature_names=[f'X{j+1}' for j in range(p)]
    def crit(m):
        if criterion=='aic': return m.aic
        if criterion=='bic': return m.bic
        return -m.adj_r2_
    # Null model AIC: n*log(SS_total/n) + 2*(0+2)  (intercept only)
    _ybar=y.mean(); _rss0=np.sum((y-_ybar)**2)
    _null_aic=n*np.log(_rss0/n)+2*2 if criterion=='aic' else n*np.log(_rss0/n)+np.log(n)*2 if criterion=='bic' else -1.0
    best=_null_aic; active=[]; remaining=list(range(p)); history=[]
    while remaining:
        best_j=None; best_c=np.inf; best_m_=None
        for j in remaining:
            cols=active+[j]; m_=LinearRegression(feature_names=[feature_names[k] for k in cols]).fit(X[:,cols],y)
            c=crit(m_)
            if c<best_c: best_c=c; best_j=j; best_m_=m_
        if best_c<best-1e-8:
            active.append(best_j); remaining.remove(best_j); best=best_c
            history.append({'step':len(active),'added':feature_names[best_j],criterion:best_c})
            if verbose: print(f"  Step {len(active):2d}: add {feature_names[best_j]:12s}  {criterion}={best_c:.4f}")
        else: break
    if not active: return [], null_m, history
    final=LinearRegression(feature_names=[feature_names[k] for k in active]).fit(X[:,active],y)
    return active, final, history


def backward_elimination(X,y,criterion='aic',pvalue_threshold=None,feature_names=None,verbose=False):
    """Backward elimination: remove one predictor at a time."""
    X=np.asarray(X,dtype=np.float64)
    if X.ndim==1: X=X.reshape(-1,1)
    n,p=X.shape
    if feature_names is None: feature_names=[f'X{j+1}' for j in range(p)]
    def crit(m):
        if criterion=='aic': return m.aic
        if criterion=='bic': return m.bic
        return -m.adj_r2_
    active=list(range(p)); history=[]
    while len(active)>1:
        cur=LinearRegression(feature_names=[feature_names[k] for k in active]).fit(X[:,active],y)
        if pvalue_threshold is not None:
            pv=cur.p_value_[1:]; worst=int(np.argmax(pv))
            if pv[worst]>pvalue_threshold:
                rem=active[worst]; history.append({'removed':feature_names[rem],'p_value':pv[worst]})
                if verbose: print(f"  Remove {feature_names[rem]:12s}  p={pv[worst]:.4f}")
                active.remove(rem)
            else: break
        else:
            base=crit(cur); best_c=base; best_rem=None
            for j in active:
                cand=[k for k in active if k!=j]
                m_=LinearRegression(feature_names=[feature_names[k] for k in cand]).fit(X[:,cand],y)
                c=crit(m_)
                if c<best_c: best_c=c; best_rem=j
            if best_rem is not None:
                history.append({'removed':feature_names[best_rem],criterion:best_c})
                if verbose: print(f"  Remove {feature_names[best_rem]:12s}  {criterion}={best_c:.4f}")
                active.remove(best_rem)
            else: break
    final=LinearRegression(feature_names=[feature_names[k] for k in active]).fit(X[:,active],y)
    return active, final, history


def stepwise_selection(X,y,criterion='aic',feature_names=None,verbose=False):
    """Mixed stepwise: forward + backward at each step (ISLP §3.2.2)."""
    X=np.asarray(X,dtype=np.float64)
    if X.ndim==1: X=X.reshape(-1,1)
    n,p=X.shape
    if feature_names is None: feature_names=[f'X{j+1}' for j in range(p)]
    def crit(m):
        if criterion=='aic': return m.aic
        if criterion=='bic': return m.bic
        return -m.adj_r2_
    # Null model AIC: n*log(SS_total/n) + 2*(0+2)  (intercept only)
    _ybar=y.mean(); _rss0=np.sum((y-_ybar)**2)
    _null_aic=n*np.log(_rss0/n)+2*2 if criterion=='aic' else n*np.log(_rss0/n)+np.log(n)*2 if criterion=='bic' else -1.0
    best=_null_aic; active=[]; remaining=list(range(p)); history=[]
    for _ in range(p*2):
        improved=False
        # Forward step
        for j in remaining:
            cols=active+[j]; m_=LinearRegression(feature_names=[feature_names[k] for k in cols]).fit(X[:,cols],y)
            c=crit(m_)
            if c<best-1e-8:
                best=c; active.append(j); remaining.remove(j)
                history.append({'action':'add','var':feature_names[j],criterion:c})
                if verbose: print(f"  Add    {feature_names[j]:12s}  {criterion}={c:.4f}")
                improved=True; break
        # Backward step
        if len(active)>1:
            for j in list(active):
                cand=[k for k in active if k!=j]; m_=LinearRegression(feature_names=[feature_names[k] for k in cand]).fit(X[:,cand],y)
                c=crit(m_)
                if c<best-1e-8:
                    best=c; active.remove(j); remaining.append(j)
                    history.append({'action':'remove','var':feature_names[j],criterion:c})
                    if verbose: print(f"  Remove {feature_names[j]:12s}  {criterion}={c:.4f}")
                    improved=True; break
        if not improved: break
    if not active: return [], LinearRegression().fit(np.ones((n,1)),y), history
    final=LinearRegression(feature_names=[feature_names[k] for k in active]).fit(X[:,active],y)
    return active, final, history


# =====================================================================
# Feature engineering helpers  (ISLP §3.3.1 and §3.3.2)
# =====================================================================

def make_interaction(X,i,j):
    """Add X[:,i]*X[:,j] interaction column (ISLP §3.3.2 Eq 3.31)."""
    X=np.asarray(X,dtype=np.float64)
    return np.hstack([X,(X[:,i]*X[:,j]).reshape(-1,1)])

def add_interactions(X,pairs=None):
    """Add all pairwise (or specified) interaction columns."""
    X=np.asarray(X,dtype=np.float64); p=X.shape[1]
    if pairs is None: pairs=[(i,j) for i in range(p) for j in range(i+1,p)]
    for i,j in pairs: X=np.hstack([X,(X[:,i]*X[:,j]).reshape(-1,1)])
    return X

def make_polynomial(x,degree):
    """
    Polynomial basis [x, x^2, ..., x^d] for polynomial regression
    (ISLP §3.3.2 Eq 3.36).
    """
    x=np.asarray(x,dtype=np.float64).ravel()
    return np.column_stack([x**d for d in range(1,degree+1)])

def dummy_encode(labels,drop_first=True):
    """
    One-hot encode a categorical predictor (ISLP §3.3.1 Eq 3.26-3.30).
    drop_first=True drops the reference category to avoid collinearity.
    Returns (X_dummy, col_names).
    """
    labels=np.asarray(labels); unique=sorted(set(labels))
    if drop_first: unique=unique[1:]
    X_d=np.zeros((len(labels),len(unique)),dtype=np.float64)
    for c,cat in enumerate(unique): X_d[:,c]=(labels==cat).astype(float)
    return X_d,[str(u) for u in unique]

def correlation_matrix(X,feature_names=None):
    """
    Print predictor correlation matrix (ISLP §3.2.2 Table 3.5).
    High correlations signal potential collinearity.
    """
    X=np.asarray(X,dtype=np.float64); p=X.shape[1]
    if feature_names is None: feature_names=[f'X{j+1}' for j in range(p)]
    C=np.corrcoef(X,rowvar=False)
    w=max(len(n) for n in feature_names)+2
    print("  "+"".join(f"{'':{w}}")+("".join(f"{nm:>{w}}" for nm in feature_names)))
    for i in range(p):
        print(f"  {feature_names[i]:>{w}}"+"".join(f"{C[i,j]:>{w}.3f}" for j in range(p)))
    return C
