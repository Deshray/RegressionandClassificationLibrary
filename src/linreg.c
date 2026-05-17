/*
 * linreg.c
 * --------
 * Full OLS linear regression following ISLP Chapter 3.
 *
 * Step-by-step derivation for linreg_fit:
 *
 *  1. Build full design matrix  Xf = [1 | X]  (n x p+1)
 *  2. Compute  XtX = Xf' Xf  ((p+1) x (p+1))
 *  3. Compute  Xty = Xf' y   (p+1)
 *  4. Invert XtX  via Gauss-Jordan  ->  XtXinv
 *  5. beta_hat = XtXinv @ Xty          [ISLP normal equations]
 *  6. y_hat    = Xf @ beta_hat
 *  7. residuals e = y - y_hat
 *  8. RSS      = e'e                    [ISLP Eq 3.16]
 *  9. TSS      = sum (y_i - y_bar)^2   [ISLP Eq 3.17]
 * 10. RSE      = sqrt(RSS / (n-p-1))   [ISLP Eq 3.15]
 * 11. R^2      = 1 - RSS/TSS           [ISLP Eq 3.17]
 * 12. Adj R^2  = 1 - (RSS/(n-p-1))/(TSS/(n-1))
 * 13. Var_j    = RSE^2 * XtXinv[j,j]  [ISLP p.74]
 * 14. SE_j     = sqrt(Var_j)
 * 15. t_j      = beta_hat_j / SE_j    [ISLP Eq 3.14]
 * 16. p_j      = 2*P(T_{n-p-1} > |t_j|)
 * 17. F-stat   = ((TSS-RSS)/p) / (RSS/(n-p-1))   [ISLP Eq 3.23]
 * 18. p_F      = P(F_{p,n-p-1} > F-stat)
 * 19. CI_j     = [beta_j - t*SE_j,  beta_j + t*SE_j]  where t=t_{n-p-1,0.025}
 */

#include "linreg.h"
#include "linalg.h"
#include "stat_dist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* linreg_fit                                                           */
/* ------------------------------------------------------------------ */
LinRegResult *linreg_fit(const double *X, const double *y, int n, int p)
{
    int q = p + 1;   /* number of columns in full design matrix (incl. intercept) */

    /* Allocate result */
    LinRegResult *r = (LinRegResult *)calloc(1, sizeof(LinRegResult));
    if (!r) return NULL;
    r->n      = n;
    r->p      = p;
    r->beta   = (double *)calloc((size_t)q, sizeof(double));
    r->se     = (double *)calloc((size_t)q, sizeof(double));
    r->t_stat = (double *)calloc((size_t)q, sizeof(double));
    r->p_value= (double *)calloc((size_t)q, sizeof(double));
    r->ci_lower=(double *)calloc((size_t)q, sizeof(double));
    r->ci_upper=(double *)calloc((size_t)q, sizeof(double));
    r->XtXinv = (double *)calloc((size_t)q * (size_t)q, sizeof(double));
    if (!r->beta || !r->se || !r->t_stat || !r->p_value ||
        !r->ci_lower || !r->ci_upper || !r->XtXinv) {
        linreg_free(r);
        return NULL;
    }

    /* ---- Step 1: Build full design matrix Xf = [1 | X]  (n x q) ---- */
    double *Xf = (double *)malloc((size_t)n * (size_t)q * sizeof(double));
    if (!Xf) { linreg_free(r); return NULL; }
    for (int i = 0; i < n; i++) {
        Xf[i*q + 0] = 1.0;                 /* intercept column */
        for (int j = 0; j < p; j++)
            Xf[i*q + j + 1] = X[i*p + j]; /* predictor columns */
    }

    /* ---- Step 2: XtX = Xf' @ Xf  (q x q) ---- */
    double *XfT = (double *)malloc((size_t)q * (size_t)n * sizeof(double));
    double *XtX = (double *)malloc((size_t)q * (size_t)q * sizeof(double));
    if (!XfT || !XtX) { free(Xf); free(XfT); free(XtX); linreg_free(r); return NULL; }

    mat_transpose(Xf, XfT, n, q);
    mat_mul(XfT, Xf, XtX, q, n, q);

    /* ---- Step 3: Xty = Xf' @ y  (q) ---- */
    double *Xty = (double *)malloc((size_t)q * sizeof(double));
    if (!Xty) { free(Xf); free(XfT); free(XtX); linreg_free(r); return NULL; }
    mat_vec_mul(XfT, y, Xty, q, n);

    /* ---- Step 4: Invert XtX, store in r->XtXinv ---- */
    mat_copy(XtX, r->XtXinv, q);          /* copy before inverting in-place */
    if (mat_inv(r->XtXinv, q) != 0) {
        /* singular — cannot fit model */
        free(Xf); free(XfT); free(XtX); free(Xty);
        linreg_free(r);
        return NULL;
    }

    /* ---- Step 5: beta_hat = XtXinv @ Xty ---- */
    mat_vec_mul(r->XtXinv, Xty, r->beta, q, q);

    /* ---- Step 6: y_hat = Xf @ beta_hat  (n) ---- */
    double *y_hat = (double *)malloc((size_t)n * sizeof(double));
    if (!y_hat) { free(Xf); free(XfT); free(XtX); free(Xty); linreg_free(r); return NULL; }
    mat_vec_mul(Xf, r->beta, y_hat, n, q);

    /* ---- Step 7-8: residuals and RSS ---- */
    double rss = 0.0;
    for (int i = 0; i < n; i++) {
        double ei = y[i] - y_hat[i];
        rss += ei * ei;
    }

    /* ---- Step 9: TSS = sum (y_i - y_bar)^2 ---- */
    double y_bar = 0.0;
    for (int i = 0; i < n; i++) y_bar += y[i];
    y_bar /= (double)n;

    double tss = 0.0;
    for (int i = 0; i < n; i++) {
        double d = y[i] - y_bar;
        tss += d * d;
    }

    /* ---- Steps 10-12: RSE, R^2, Adjusted R^2 ---- */
    int df_resid = n - p - 1;             /* n - p - 1  (ISLP p.74) */
    r->rse       = sqrt(rss / (double)df_resid);
    r->r_squared = (tss > 0.0) ? 1.0 - rss / tss : 0.0;
    r->adj_r2    = (tss > 0.0)
                   ? 1.0 - (rss / (double)df_resid) / (tss / (double)(n - 1))
                   : 0.0;

    /* ---- Steps 13-16: SE, t-stats, p-values ---- */
    double rse2     = r->rse * r->rse;
    double t_crit   = t_quantile(0.975, (double)df_resid);  /* t_{df, 0.025} */

    for (int j = 0; j < q; j++) {
        r->se[j]     = sqrt(rse2 * r->XtXinv[j*q + j]);
        r->t_stat[j] = (r->se[j] > 1e-16) ? r->beta[j] / r->se[j] : 0.0;
        r->p_value[j]= t_pvalue_two_sided(r->t_stat[j], (double)df_resid);
        r->ci_lower[j]= r->beta[j] - t_crit * r->se[j];
        r->ci_upper[j]= r->beta[j] + t_crit * r->se[j];
    }

    /* ---- Steps 17-18: F-statistic and its p-value (ISLP Eq 3.23) ----
     *
     *  H0: beta_1 = beta_2 = ... = beta_p = 0  (intercept unconstrained)
     *  F = ((TSS - RSS) / p) / (RSS / (n-p-1))
     *
     *  Under H0, F ~ F(p, n-p-1).
     */
    if (p > 0 && df_resid > 0) {
        r->f_stat   = ((tss - rss) / (double)p) / (rss / (double)df_resid);
        r->f_pvalue = f_pvalue(r->f_stat, p, df_resid);
    } else {
        r->f_stat   = 0.0;
        r->f_pvalue = 1.0;
    }

    /* Clean up temporaries */
    free(Xf); free(XfT); free(XtX); free(Xty); free(y_hat);
    return r;
}

/* ------------------------------------------------------------------ */
/* linreg_free                                                          */
/* ------------------------------------------------------------------ */
void linreg_free(LinRegResult *r)
{
    if (!r) return;
    free(r->beta);    free(r->se);     free(r->t_stat);
    free(r->p_value); free(r->ci_lower); free(r->ci_upper);
    free(r->XtXinv);
    free(r);
}

/* ------------------------------------------------------------------ */
/* Internal: build full predictor row x_0 = [1, x_new_row]            */
/* ------------------------------------------------------------------ */
static void build_x0(const double *x_new_row, int p, double *x0)
{
    x0[0] = 1.0;
    for (int j = 0; j < p; j++)
        x0[j + 1] = x_new_row[j];
}

/* ------------------------------------------------------------------ */
/* linreg_predict  y_hat = Xnew @ beta_hat                            */
/* ------------------------------------------------------------------ */
void linreg_predict(const LinRegResult *r, const double *X_new,
                    double *y_hat, int m)
{
    int q = r->p + 1;
    double *x0 = (double *)malloc((size_t)q * sizeof(double));
    if (!x0) return;

    for (int i = 0; i < m; i++) {
        build_x0(X_new + i * r->p, r->p, x0);
        y_hat[i] = vec_dot(x0, r->beta, q);
    }
    free(x0);
}

/* ------------------------------------------------------------------ */
/* Internal: half-width for CI / PI at a single new point x0          */
/*                                                                      */
/* leverage  h = x0' (X'X)^{-1} x0                                    */
/* CI half-width = t_crit * RSE * sqrt(h)                              */
/* PI half-width = t_crit * RSE * sqrt(1 + h)                         */
/* ------------------------------------------------------------------ */
static double leverage(const LinRegResult *r, const double *x0)
{
    int q = r->p + 1;
    /* tmp = XtXinv @ x0  (q) */
    double *tmp = (double *)malloc((size_t)q * sizeof(double));
    if (!tmp) return 0.0;
    mat_vec_mul(r->XtXinv, x0, tmp, q, q);
    double h = vec_dot(x0, tmp, q);
    free(tmp);
    return h;
}

/* ------------------------------------------------------------------ */
/* linreg_predict_ci  — confidence interval for E[Y|x_0]              */
/* ------------------------------------------------------------------ */
void linreg_predict_ci(const LinRegResult *r,
                       const double *X_new, int m, double alpha,
                       double *lower, double *upper)
{
    int    q      = r->p + 1;
    int    df     = r->n - r->p - 1;
    double t_crit = t_quantile(1.0 - alpha / 2.0, (double)df);
    double *x0    = (double *)malloc((size_t)q * sizeof(double));
    if (!x0) return;

    for (int i = 0; i < m; i++) {
        build_x0(X_new + i * r->p, r->p, x0);
        double y_hat_i = vec_dot(x0, r->beta, q);
        double h       = leverage(r, x0);
        double half    = t_crit * r->rse * sqrt(h);
        lower[i] = y_hat_i - half;
        upper[i] = y_hat_i + half;
    }
    free(x0);
}

/* ------------------------------------------------------------------ */
/* linreg_predict_pi  — prediction interval for individual Y           */
/*                                                                      */
/* PI is wider than CI because it adds the irreducible error variance  */
/* (the +1 inside the sqrt), reflecting uncertainty in the individual  */
/* response beyond the uncertainty in the mean (ISLP 3.2.2).          */
/* ------------------------------------------------------------------ */
void linreg_predict_pi(const LinRegResult *r,
                       const double *X_new, int m, double alpha,
                       double *lower, double *upper)
{
    int    q      = r->p + 1;
    int    df     = r->n - r->p - 1;
    double t_crit = t_quantile(1.0 - alpha / 2.0, (double)df);
    double *x0    = (double *)malloc((size_t)q * sizeof(double));
    if (!x0) return;

    for (int i = 0; i < m; i++) {
        build_x0(X_new + i * r->p, r->p, x0);
        double y_hat_i = vec_dot(x0, r->beta, q);
        double h       = leverage(r, x0);
        double half    = t_crit * r->rse * sqrt(1.0 + h);  /* +1 is irreducible */
        lower[i] = y_hat_i - half;
        upper[i] = y_hat_i + half;
    }
    free(x0);
}
