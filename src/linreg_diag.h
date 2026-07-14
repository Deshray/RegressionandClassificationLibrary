/*
 * linreg_diag.h
 * -------------
 * Regression diagnostic statistics from ISLP Section 3.3.3 "Potential Problems".
 *
 * ISLP identifies six potential problems:
 *   1. Non-linearity            — inspect residual vs fitted plot
 *   2. Correlated error terms   — inspect residuals vs index (time-series)
 *   3. Non-constant variance    — inspect scale-location (sqrt|e| vs fitted)
 *   4. Outliers                 — studentized residuals  |r_i| > 3
 *   5. High-leverage points     — leverage h_i >> (p+1)/n
 *   6. Collinearity             — VIF (in linreg_select.h)
 *
 * Formulas
 * --------
 * Hat matrix diagonal (leverage):
 *   h_i = x_i' (X'X)^{-1} x_i        [ISLP Eq 3.37, p.98]
 *   where x_i = [1, x_{i1}, ..., x_{ip}] is the full design row.
 *   Average leverage = (p+1)/n.
 *   Rule of thumb: h_i > 2*(p+1)/n is "high leverage".
 *
 * Internally studentized residuals:
 *   r_i = e_i / (RSE * sqrt(1 - h_i))   [ISLP p.97]
 *   |r_i| > 3 → possible outlier.
 *
 * Externally studentized residuals (delete-one):
 *   t_i = e_i / (RSE_{-i} * sqrt(1 - h_i))
 *   RSE_{-i}^2 = ((n-p-2)*RSE^2 - e_i^2/(1-h_i)) / (n-p-2)
 *   Under H0: t_i ~ t_{n-p-2}.
 *
 * Cook's distance:
 *   D_i = e_i^2 * h_i / ((p+1) * RSE^2 * (1-h_i)^2)
 *   D_i > 1 → influential observation.
 *   Equivalent to: D_i = r_i^2 * h_i / ((p+1)*(1-h_i))
 *
 * DFFITS (difference in fits):
 *   DFFITS_i = t_i * sqrt(h_i / (1-h_i))
 *   |DFFITS| > 2*sqrt((p+1)/n) → influential.
 */

#ifndef LINREG_DIAG_H
#define LINREG_DIAG_H

#include "linreg.h"

/* ------------------------------------------------------------------ */
/* Diagnostics result structure                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;
    int     p;

    double *residuals;         /* raw residuals e_i = y_i - y_hat_i  (n)    */
    double *leverage;          /* hat diagonal h_i  (n)                       */
    double *std_residuals;     /* internally studentized r_i  (n)             */
    double *ext_studentized;   /* externally studentized t_i  (n)             */
    double *cooks_d;           /* Cook's distance D_i  (n)                    */
    double *dffits;            /* DFFITS_i  (n)                               */

    double  avg_leverage;      /* (p+1)/n — threshold baseline                */
    int    *outlier_flag;      /* 1 if |ext_studentized| > 3, else 0  (n)    */
    int    *high_leverage_flag;/* 1 if h_i > 2*(p+1)/n, else 0  (n)          */
    int    *influential_flag;  /* 1 if cooks_d > 1, else 0  (n)              */

} LinRegDiag;

/* ------------------------------------------------------------------ */
/* API                                                                  */
/* ------------------------------------------------------------------ */

/*
 * Compute all regression diagnostics.
 *   r   : fitted LinRegResult from linreg_fit().
 *   X   : (n x p) original predictor matrix (no intercept column).
 *   y   : (n) response vector.
 * Returns heap-allocated LinRegDiag.  Caller must call linreg_diag_free().
 */
LinRegDiag *linreg_diagnostics(const LinRegResult *r,
                                const double *X, const double *y,
                                int n);

void linreg_diag_free(LinRegDiag *d);

#endif /* LINREG_DIAG_H */
