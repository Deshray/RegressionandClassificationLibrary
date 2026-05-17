/*
 * linreg.h
 * --------
 * Ordinary Least Squares linear regression following ISLP Chapter 3.
 *
 * Implements the full statistical methodology from Chapter 3:
 *
 *  beta_hat = (X'X)^{-1} X'y                             [normal equations]
 *  RSS      = sum (y_i - y_hat_i)^2                       [Eq 3.3 / 3.16]
 *  TSS      = sum (y_i - y_bar)^2                         [Eq 3.17]
 *  RSE      = sqrt( RSS / (n-p-1) )                       [Eq 3.15]
 *  R^2      = 1 - RSS/TSS                                 [Eq 3.17]
 *  Adj R^2  = 1 - (RSS/(n-p-1)) / (TSS/(n-1))            [ISLP p.86]
 *  Var(beta)= RSE^2 * (X'X)^{-1}                         [ISLP p.74]
 *  SE(beta_j)= sqrt( Var(beta)[j,j] )                    [ISLP p.74]
 *  t_j      = beta_hat_j / SE(beta_hat_j)                [Eq 3.14]
 *  p_j      = 2 * P(T_{n-p-1} > |t_j|)                  [ISLP p.76]
 *  F-stat   = ((TSS-RSS)/p) / (RSS/(n-p-1))              [Eq 3.23]
 *  p_F      = P(F_{p,n-p-1} > F-stat)                    [ISLP p.84]
 *  95% CI   = beta_hat_j +/- t_{n-p-1,0.025}*SE(beta_j) [Eq 3.9-3.11]
 *
 * Prediction intervals (ISLP 3.2.2):
 *  y_hat_0 +/- t_{n-p-1,0.025} * RSE * sqrt(1 + x_0'(X'X)^{-1}x_0)
 *
 * Confidence intervals for the mean response (ISLP 3.2.2):
 *  y_hat_0 +/- t_{n-p-1,0.025} * RSE * sqrt(x_0'(X'X)^{-1}x_0)
 */

#ifndef LINREG_H
#define LINREG_H

/* ------------------------------------------------------------------ */
/* Result structure                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;           /* number of observations                      */
    int     p;           /* number of predictors (excluding intercept)  */

    /* Coefficient estimates — length (p+1), index 0 is intercept */
    double *beta;        /* beta_hat[j]  (ISLP Eq 3.4 / matrix form)   */
    double *se;          /* SE(beta_hat[j])  (ISLP p.74)                */
    double *t_stat;      /* t_j = beta_hat[j] / SE[j]  (ISLP Eq 3.14)  */
    double *p_value;     /* 2*P(T_{n-p-1} > |t_j|)  (ISLP p.76)       */
    double *ci_lower;    /* 95% CI lower: beta_hat - t*SE               */
    double *ci_upper;    /* 95% CI upper: beta_hat + t*SE               */

    /* Model-level statistics */
    double  rse;         /* Residual Standard Error  (ISLP Eq 3.15)     */
    double  r_squared;   /* R^2  (ISLP Eq 3.17)                         */
    double  adj_r2;      /* Adjusted R^2  (ISLP p.86)                   */
    double  f_stat;      /* F-statistic  (ISLP Eq 3.23)                 */
    double  f_pvalue;    /* P(F_{p, n-p-1} > F-stat)                   */

    /*
     * (X'X)^{-1} stored flat row-major, size (p+1)^2.
     * Needed internally for confidence / prediction intervals.
     * Not exposed in the Python summary but used by linreg_predict_ci
     * and linreg_predict_pi.
     */
    double *XtXinv;

} LinRegResult;

/* ------------------------------------------------------------------ */
/* API                                                                  */
/* ------------------------------------------------------------------ */

/*
 * Fit OLS linear regression.
 *
 *   X  : (n x p) design matrix WITHOUT intercept column (raw predictors).
 *          An intercept column of 1s is prepended internally so that
 *          the full design matrix is [1 | X] of shape (n x p+1).
 *   y  : (n) response vector.
 *   n  : number of observations.
 *   p  : number of predictors (columns of X).
 *
 * Returns a heap-allocated LinRegResult.  Caller must call linreg_free().
 * Returns NULL on failure (singular X'X, allocation error).
 */
LinRegResult *linreg_fit(const double *X, const double *y, int n, int p);

/*
 * Free a LinRegResult previously returned by linreg_fit.
 */
void linreg_free(LinRegResult *r);

/*
 * Point predictions  y_hat = X_new @ beta_hat  (with intercept prepended).
 *
 *   X_new  : (m x p) matrix of new predictor values (no intercept column).
 *   y_hat  : (m) output — caller allocates.
 *   m      : number of new observations.
 */
void linreg_predict(const LinRegResult *r, const double *X_new,
                    double *y_hat, int m);

/*
 * Confidence interval for the mean response at each row of X_new.
 * (ISLP 3.2.2 — reducible error only)
 *
 *   lower, upper : (m) output vectors — caller allocates.
 *   alpha        : significance level (e.g. 0.05 for 95% CI).
 *
 * CI:  y_hat_0 +/- t_{n-p-1, alpha/2} * RSE * sqrt(x_0'(X'X)^{-1}x_0)
 */
void linreg_predict_ci(const LinRegResult *r,
                       const double *X_new, int m, double alpha,
                       double *lower, double *upper);

/*
 * Prediction interval for an individual response at each row of X_new.
 * (ISLP 3.2.2 — reducible + irreducible error)
 *
 *   lower, upper : (m) output vectors — caller allocates.
 *   alpha        : significance level (e.g. 0.05 for 95% PI).
 *
 * PI:  y_hat_0 +/- t_{n-p-1, alpha/2} * RSE * sqrt(1 + x_0'(X'X)^{-1}x_0)
 */
void linreg_predict_pi(const LinRegResult *r,
                       const double *X_new, int m, double alpha,
                       double *lower, double *upper);

#endif /* LINREG_H */
