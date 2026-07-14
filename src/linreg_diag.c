/*
 * linreg_diag.c
 * -------------
 * Regression diagnostics: ISLP Section 3.3.3.
 */

#include "linreg_diag.h"
#include "linalg.h"
#include "stat_dist.h"
#include <stdlib.h>
#include <math.h>

LinRegDiag *linreg_diagnostics(const LinRegResult *r,
                                const double *X, const double *y,
                                int n)
{
    int p = r->p;
    int q = p + 1;   /* full design width (intercept + p predictors) */

    LinRegDiag *d = (LinRegDiag *)calloc(1, sizeof(LinRegDiag));
    if (!d) return NULL;
    d->n = n; d->p = p;

    d->residuals          = (double *)malloc((size_t)n * sizeof(double));
    d->leverage           = (double *)malloc((size_t)n * sizeof(double));
    d->std_residuals      = (double *)malloc((size_t)n * sizeof(double));
    d->ext_studentized    = (double *)malloc((size_t)n * sizeof(double));
    d->cooks_d            = (double *)malloc((size_t)n * sizeof(double));
    d->dffits             = (double *)malloc((size_t)n * sizeof(double));
    d->outlier_flag       = (int    *)malloc((size_t)n * sizeof(int));
    d->high_leverage_flag = (int    *)malloc((size_t)n * sizeof(int));
    d->influential_flag   = (int    *)malloc((size_t)n * sizeof(int));

    if (!d->residuals || !d->leverage || !d->std_residuals ||
        !d->ext_studentized || !d->cooks_d || !d->dffits ||
        !d->outlier_flag || !d->high_leverage_flag || !d->influential_flag) {
        linreg_diag_free(d); return NULL;
    }

    double avg_lev = (double)q / (double)n;
    d->avg_leverage = avg_lev;

    double rse2 = r->rse * r->rse;
    int    df   = n - p - 1;   /* residual df */

    /* tmp vector for matrix-vector products (length q) */
    double *tmp = (double *)malloc((size_t)q * sizeof(double));
    double *x0  = (double *)malloc((size_t)q * sizeof(double));
    if (!tmp || !x0) { free(tmp); free(x0); linreg_diag_free(d); return NULL; }

    for (int i = 0; i < n; i++) {
        /* Build full design row  x0 = [1, x_{i1}, ..., x_{ip}] */
        x0[0] = 1.0;
        for (int j = 0; j < p; j++) x0[j+1] = X[i*p + j];

        /* ---- Residual  e_i = y_i - x_i' beta ---- */
        double y_hat_i = 0.0;
        for (int j = 0; j < q; j++) y_hat_i += x0[j] * r->beta[j];
        double ei = y[i] - y_hat_i;
        d->residuals[i] = ei;

        /* ---- Leverage  h_i = x_i' (X'X)^{-1} x_i ----
         * r->XtXinv is (q x q) stored row-major.
         * tmp = XtXinv @ x0
         */
        mat_vec_mul(r->XtXinv, x0, tmp, q, q);
        double hi = vec_dot(x0, tmp, q);
        /* clamp to valid range to guard against floating-point noise */
        hi = (hi < 1e-15) ? 1e-15 : (hi > 1.0 - 1e-15 ? 1.0 - 1e-15 : hi);
        d->leverage[i] = hi;

        /* ---- Internally studentized residual  r_i = e_i / (RSE * sqrt(1-h_i)) ---- */
        double denom_int = r->rse * sqrt(1.0 - hi);
        d->std_residuals[i] = (denom_int > 1e-16) ? ei / denom_int : 0.0;

        /* ---- Externally studentized (delete-one) RSE
         *   RSE_{-i}^2 = ((df-1)*RSE^2 - e_i^2/(1-h_i)) / (df-1)
         * This uses the shortcut formula to avoid actually refitting.
         */
        double rse2_minus_i = -1.0;
        if (df > 1) {
            double num = (double)(df - 1) * rse2 - ei * ei / (1.0 - hi);
            if (num > 0.0) rse2_minus_i = num / (double)(df - 1);
        }
        double ti;
        if (rse2_minus_i > 0.0) {
            double denom_ext = sqrt(rse2_minus_i * (1.0 - hi));
            ti = (denom_ext > 1e-16) ? ei / denom_ext : 0.0;
        } else {
            ti = d->std_residuals[i];   /* fallback */
        }
        d->ext_studentized[i] = ti;

        /* ---- Cook's distance
         *   D_i = e_i^2 * h_i / ((p+1) * RSE^2 * (1-h_i)^2)
         *       = r_i^2 * h_i / ((p+1) * (1-h_i))
         */
        double one_minus_h = 1.0 - hi;
        double ri = d->std_residuals[i];
        d->cooks_d[i] = (q > 0 && one_minus_h > 1e-15 && rse2 > 1e-20)
            ? ri * ri * hi / ((double)q * one_minus_h)
            : 0.0;

        /* ---- DFFITS_i = t_i * sqrt(h_i / (1-h_i)) ---- */
        d->dffits[i] = ti * sqrt(hi / one_minus_h);

        /* ---- Flags ---- */
        d->outlier_flag[i]        = (fabs(ti)  > 3.0)                 ? 1 : 0;
        d->high_leverage_flag[i]  = (hi        > 2.0 * avg_lev)       ? 1 : 0;
        d->influential_flag[i]    = (d->cooks_d[i] > 1.0)             ? 1 : 0;
    }

    free(tmp); free(x0);
    return d;
}

void linreg_diag_free(LinRegDiag *d)
{
    if (!d) return;
    free(d->residuals);     free(d->leverage);    free(d->std_residuals);
    free(d->ext_studentized); free(d->cooks_d);  free(d->dffits);
    free(d->outlier_flag);  free(d->high_leverage_flag);
    free(d->influential_flag);
    free(d);
}
