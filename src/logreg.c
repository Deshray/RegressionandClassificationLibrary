/*
 * logreg.c
 * --------
 * Binary logistic regression via IRLS (Newton-Raphson), ISLP Chapter 4.
 *
 * IRLS derivation
 * ---------------
 * The log-likelihood for binary logistic regression is:
 *
 *   ell(beta) = sum_i [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
 *
 * where  p_i = sigmoid(x_i' beta) = 1 / (1 + exp(-x_i' beta)).
 *
 * The score (gradient):
 *   del ell / del beta = X' (y - mu)          where  mu_i = p_i
 *
 * The Hessian (negative):
 *   -del^2 ell / del beta del beta' = X' W X   where  W = diag(mu*(1-mu))
 *
 * Newton-Raphson update:
 *   beta_new = beta + (X'WX)^{-1} X'(y-mu)
 *            = (X'WX)^{-1} X'Wz
 *
 * where the working response  z = X*beta + W^{-1}(y-mu)  is a weighted
 * adjusted dependent variable (Eq derivable from ISLP GLM framework §4.6).
 *
 * After convergence the Fisher information matrix I(beta) = X'WX, so
 *   Cov(beta_hat) = I(beta_hat)^{-1} = (X'WX)^{-1}
 * and standard errors follow from the diagonal (ISLP 4.3.2).
 */

#include "logreg.h"
#include "linalg.h"
#include "stat_dist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Internal: sigmoid  1 / (1 + exp(-x))                               */
/* Numerically stable: avoids overflow for very negative x.           */
/* ------------------------------------------------------------------ */
static double sigmoid(double x)
{
    if (x >= 0.0)
        return 1.0 / (1.0 + exp(-x));
    else {
        double ex = exp(x);
        return ex / (1.0 + ex);
    }
}

/* ------------------------------------------------------------------ */
/* Internal: log-likelihood                                             */
/* ------------------------------------------------------------------ */
static double log_likelihood(const double *mu, const double *y, int n)
{
    double ll = 0.0;
    const double EPS = 1e-15;
    for (int i = 0; i < n; i++) {
        double p = mu[i];
        p = (p < EPS) ? EPS : (p > 1.0 - EPS ? 1.0 - EPS : p);
        ll += y[i] * log(p) + (1.0 - y[i]) * log(1.0 - p);
    }
    return ll;
}

/* ------------------------------------------------------------------ */
/* logreg_fit                                                           */
/* ------------------------------------------------------------------ */
LogRegResult *logreg_fit(const double *X, const double *y, int n, int p,
                         int max_iter, double tol)
{
    int q = p + 1;   /* full model dimension (incl. intercept) */

    /* Allocate result */
    LogRegResult *r = (LogRegResult *)calloc(1, sizeof(LogRegResult));
    if (!r) return NULL;
    r->n      = n;
    r->p      = p;
    r->beta   = (double *)calloc((size_t)q, sizeof(double));
    r->se     = (double *)calloc((size_t)q, sizeof(double));
    r->z_stat = (double *)calloc((size_t)q, sizeof(double));
    r->p_value= (double *)calloc((size_t)q, sizeof(double));
    if (!r->beta || !r->se || !r->z_stat || !r->p_value) {
        logreg_free(r);
        return NULL;
    }

    /* Build full design matrix  Xf = [1 | X]  (n x q) */
    double *Xf = (double *)malloc((size_t)n * (size_t)q * sizeof(double));
    if (!Xf) { logreg_free(r); return NULL; }
    for (int i = 0; i < n; i++) {
        Xf[i*q] = 1.0;
        for (int j = 0; j < p; j++)
            Xf[i*q + j + 1] = X[i*p + j];
    }

    /* Working arrays */
    double *mu    = (double *)calloc((size_t)n, sizeof(double));  /* fitted probs  */
    double *eta   = (double *)calloc((size_t)n, sizeof(double));  /* linear pred   */
    double *z     = (double *)calloc((size_t)n, sizeof(double));  /* working resp  */
    double *w     = (double *)calloc((size_t)n, sizeof(double));  /* IRLS weights  */
    double *XtWX  = (double *)calloc((size_t)q * (size_t)q, sizeof(double)); /* q x q */
    double *XtWz  = (double *)calloc((size_t)q, sizeof(double));  /* q      */
    double *beta_new = (double *)calloc((size_t)q, sizeof(double));
    if (!mu || !eta || !z || !w || !XtWX || !XtWz || !beta_new) {
        free(Xf); free(mu); free(eta); free(z); free(w);
        free(XtWX); free(XtWz); free(beta_new);
        logreg_free(r);
        return NULL;
    }

    /* ---- Null deviance: intercept-only model ----
     *
     * For the null model, beta = (log(p_bar/(1-p_bar)), 0, ..., 0)
     * where p_bar = mean(y).  The null deviance is -2 * log-lik(null).
     */
    double y_bar = 0.0;
    for (int i = 0; i < n; i++) y_bar += y[i];
    y_bar /= (double)n;
    y_bar = (y_bar < 1e-10) ? 1e-10 : (y_bar > 1.0 - 1e-10 ? 1.0 - 1e-10 : y_bar);

    double null_ll = 0.0;
    for (int i = 0; i < n; i++) {
        const double EPS = 1e-15;
        double pm = (y_bar < EPS) ? EPS : (y_bar > 1.0-EPS ? 1.0-EPS : y_bar);
        null_ll += y[i] * log(pm) + (1.0 - y[i]) * log(1.0 - pm);
    }
    r->null_deviance = -2.0 * null_ll;

    /* ---- Initialise beta: intercept = logit(y_bar), rest = 0 ---- */
    r->beta[0] = log(y_bar / (1.0 - y_bar));
    for (int j = 1; j < q; j++) r->beta[j] = 0.0;

    /* ---- IRLS loop ---- */
    r->converged = 0;
    r->n_iter    = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        r->n_iter = iter + 1;

        /* Step A: eta = Xf @ beta  (linear predictor) */
        mat_vec_mul(Xf, r->beta, eta, n, q);

        /* Step B: mu = sigmoid(eta),  w_i = mu_i*(1-mu_i) */
        for (int i = 0; i < n; i++) {
            mu[i] = sigmoid(eta[i]);
            double wi = mu[i] * (1.0 - mu[i]);
            w[i]  = (wi < 1e-15) ? 1e-15 : wi;  /* guard against zero weight */
        }

        /* Step C: working response  z_i = eta_i + (y_i - mu_i) / w_i */
        for (int i = 0; i < n; i++)
            z[i] = eta[i] + (y[i] - mu[i]) / w[i];

        /* Step D: XtWX = Xf' @ W @ Xf  (q x q)
         * We compute it as  sum_i w_i * x_i * x_i'  to avoid forming W explicitly.
         */
        for (int a = 0; a < q * q; a++) XtWX[a] = 0.0;
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < q; a++)
                for (int b = 0; b < q; b++)
                    XtWX[a*q + b] += w[i] * Xf[i*q + a] * Xf[i*q + b];
        }

        /* Step E: XtWz = Xf' @ W @ z  (q)
         * = sum_i w_i * z_i * x_i
         */
        for (int a = 0; a < q; a++) XtWz[a] = 0.0;
        for (int i = 0; i < n; i++) {
            double wizi = w[i] * z[i];
            for (int a = 0; a < q; a++)
                XtWz[a] += wizi * Xf[i*q + a];
        }

        /* Step F: solve  (XtWX) @ beta_new = XtWz
         * by inverting XtWX (Gauss-Jordan) and multiplying.
         */
        double *XtWX_copy = (double *)malloc((size_t)q * (size_t)q * sizeof(double));
        if (!XtWX_copy) {
            /* clean up and fail gracefully */
            break;
        }
        memcpy(XtWX_copy, XtWX, (size_t)q * (size_t)q * sizeof(double));
        if (mat_inv(XtWX_copy, q) != 0) {
            free(XtWX_copy);
            break;   /* singular — stop */
        }
        mat_vec_mul(XtWX_copy, XtWz, beta_new, q, q);
        free(XtWX_copy);

        /* Step G: check convergence  max|beta_new - beta| < tol */
        double max_delta = 0.0;
        for (int j = 0; j < q; j++) {
            double d = fabs(beta_new[j] - r->beta[j]);
            if (d > max_delta) max_delta = d;
        }

        /* Update beta */
        for (int j = 0; j < q; j++) r->beta[j] = beta_new[j];

        if (max_delta < tol) {
            r->converged = 1;
            break;
        }
    }

    /* ---- After convergence: SE, z-stats, p-values ---- */
    /* Recompute final mu and final XtWX using converged beta */
    mat_vec_mul(Xf, r->beta, eta, n, q);
    for (int i = 0; i < n; i++) {
        mu[i] = sigmoid(eta[i]);
        double wi = mu[i] * (1.0 - mu[i]);
        w[i]  = (wi < 1e-15) ? 1e-15 : wi;
    }
    for (int a = 0; a < q * q; a++) XtWX[a] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int a = 0; a < q; a++)
            for (int b = 0; b < q; b++)
                XtWX[a*q + b] += w[i] * Xf[i*q + a] * Xf[i*q + b];
    }

    /* Cov(beta_hat) = (X'WX)^{-1} = Fisher information matrix inverse */
    if (mat_inv(XtWX, q) == 0) {
        for (int j = 0; j < q; j++) {
            r->se[j]     = sqrt(fabs(XtWX[j*q + j]));
            r->z_stat[j] = (r->se[j] > 1e-16) ? r->beta[j] / r->se[j] : 0.0;
            r->p_value[j]= t_pvalue_two_sided(r->z_stat[j], 1e9); /* N(0,1) limit */
        }
    }

    /* ---- Deviance and AIC ---- */
    r->deviance = -2.0 * log_likelihood(mu, y, n);
    r->aic      = r->deviance + 2.0 * (double)q;

    /* Clean up */
    free(Xf); free(mu); free(eta); free(z); free(w);
    free(XtWX); free(XtWz); free(beta_new);

    return r;
}

/* ------------------------------------------------------------------ */
/* logreg_free                                                          */
/* ------------------------------------------------------------------ */
void logreg_free(LogRegResult *r)
{
    if (!r) return;
    free(r->beta); free(r->se); free(r->z_stat); free(r->p_value);
    free(r);
}

/* ------------------------------------------------------------------ */
/* logreg_predict_prob                                                  */
/* ------------------------------------------------------------------ */
void logreg_predict_prob(const LogRegResult *r,
                         const double *X_new, double *prob, int m)
{
    for (int i = 0; i < m; i++) {
        double eta_i = r->beta[0];
        for (int j = 0; j < r->p; j++)
            eta_i += r->beta[j + 1] * X_new[i * r->p + j];
        prob[i] = sigmoid(eta_i);
    }
}

/* ------------------------------------------------------------------ */
/* logreg_predict                                                       */
/* ------------------------------------------------------------------ */
void logreg_predict(const LogRegResult *r,
                    const double *X_new, int *labels,
                    int m, double threshold)
{
    double *prob = (double *)malloc((size_t)m * sizeof(double));
    if (!prob) return;
    logreg_predict_prob(r, X_new, prob, m);
    for (int i = 0; i < m; i++)
        labels[i] = (prob[i] > threshold) ? 1 : 0;
    free(prob);
}
