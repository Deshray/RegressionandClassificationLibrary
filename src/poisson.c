/*
 * poisson.c
 * ---------
 * Poisson regression via IRLS: ISLP Section 4.6.2.
 *
 * Structurally identical to logistic regression but with:
 *   link function: g(mu) = log(mu)
 *   variance:      V(mu) = mu          (Poisson)
 *   IRLS weight:   w_i = mu_i
 *   working resp:  z_i = eta_i + (y_i - mu_i)/mu_i
 */

#include "poisson.h"
#include "linalg.h"
#include "stat_dist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

static double safe_exp(double x)
{
    if (x > 700.0)  return exp(700.0);
    if (x < -700.0) return exp(-700.0);
    return exp(x);
}

/* Poisson log-likelihood */

PoissonResult *poisson_fit(const double *X, const double *y, int n, int p,
                            int max_iter, double tol)
{
    int q = p + 1;

    PoissonResult *r = (PoissonResult *)calloc(1, sizeof(PoissonResult));
    if (!r) return NULL;
    r->n = n; r->p = p;
    r->beta    = (double *)calloc((size_t)q, sizeof(double));
    r->se      = (double *)calloc((size_t)q, sizeof(double));
    r->z_stat  = (double *)calloc((size_t)q, sizeof(double));
    r->p_value = (double *)calloc((size_t)q, sizeof(double));
    if (!r->beta||!r->se||!r->z_stat||!r->p_value){ poisson_free(r); return NULL; }

    /* Build full design matrix Xf = [1|X] (n x q) */
    double *Xf = (double *)malloc((size_t)n*(size_t)q*sizeof(double));
    if (!Xf) { poisson_free(r); return NULL; }
    for (int i=0;i<n;i++) {
        Xf[i*q]=1.0;
        for (int j=0;j<p;j++) Xf[i*q+j+1]=X[i*p+j];
    }

    /* Working arrays */
    double *mu    = (double *)calloc((size_t)n, sizeof(double));
    double *eta   = (double *)calloc((size_t)n, sizeof(double));
    double *z     = (double *)calloc((size_t)n, sizeof(double));
    double *w     = (double *)calloc((size_t)n, sizeof(double));
    double *XtWX  = (double *)calloc((size_t)q*(size_t)q, sizeof(double));
    double *XtWz  = (double *)calloc((size_t)q, sizeof(double));
    double *beta_new = (double *)calloc((size_t)q, sizeof(double));
    if (!mu||!eta||!z||!w||!XtWX||!XtWz||!beta_new) {
        free(Xf);free(mu);free(eta);free(z);free(w);
        free(XtWX);free(XtWz);free(beta_new); poisson_free(r); return NULL;
    }

    /* Null model: intercept = log(mean(y)) */
    double y_bar = 0.0;
    for (int i=0;i<n;i++) y_bar += y[i];
    y_bar /= (double)n;
    y_bar = (y_bar < 1e-10) ? 1e-10 : y_bar;
    r->beta[0] = log(y_bar);

    /* Null deviance */
    double null_mu_val = y_bar;
    double null_ll = 0.0;
    for (int i=0;i<n;i++) null_ll += y[i]*log(null_mu_val) - null_mu_val;
    r->null_deviance = -2.0*null_ll;

    /* IRLS */
    r->converged = 0;
    for (int iter=0; iter<max_iter; iter++) {
        r->n_iter = iter+1;

        /* eta = Xf @ beta, mu = exp(eta), w = mu */
        for (int i=0;i<n;i++) {
            eta[i] = 0.0;
            for (int j=0;j<q;j++) eta[i] += Xf[i*q+j]*r->beta[j];
            mu[i] = safe_exp(eta[i]);
            w[i]  = (mu[i] < 1e-10) ? 1e-10 : mu[i];
        }

        /* working response z_i = eta_i + (y_i - mu_i)/w_i */
        for (int i=0;i<n;i++) z[i] = eta[i] + (y[i]-mu[i])/w[i];

        /* XtWX = sum_i w_i x_i x_i'  (q x q) */
        for (int a=0;a<q*q;a++) XtWX[a]=0.0;
        for (int i=0;i<n;i++)
            for (int a=0;a<q;a++)
                for (int b=0;b<q;b++)
                    XtWX[a*q+b] += w[i]*Xf[i*q+a]*Xf[i*q+b];

        /* XtWz = sum_i w_i z_i x_i */
        for (int a=0;a<q;a++) XtWz[a]=0.0;
        for (int i=0;i<n;i++) {
            double wz = w[i]*z[i];
            for (int a=0;a<q;a++) XtWz[a] += wz*Xf[i*q+a];
        }

        /* Solve: beta_new = XtWX^{-1} XtWz */
        double *XtWX_copy = (double *)malloc((size_t)q*(size_t)q*sizeof(double));
        if (!XtWX_copy) break;
        memcpy(XtWX_copy, XtWX, (size_t)q*(size_t)q*sizeof(double));
        if (mat_inv(XtWX_copy,q)!=0) { free(XtWX_copy); break; }
        for (int a=0;a<q;a++) {
            beta_new[a]=0.0;
            for (int b=0;b<q;b++) beta_new[a] += XtWX_copy[a*q+b]*XtWz[b];
        }
        free(XtWX_copy);

        double max_d=0.0;
        for (int j=0;j<q;j++) { double d=fabs(beta_new[j]-r->beta[j]); if(d>max_d) max_d=d; }
        for (int j=0;j<q;j++) r->beta[j]=beta_new[j];
        if (max_d<tol) { r->converged=1; break; }
    }

    /* SE, z-stats, p-values from final XtWX */
    for (int i=0;i<n;i++) {
        eta[i]=0.0;
        for (int j=0;j<q;j++) eta[i]+=Xf[i*q+j]*r->beta[j];
        mu[i]=safe_exp(eta[i]);
        w[i]=(mu[i]<1e-10)?1e-10:mu[i];
    }
    for (int a=0;a<q*q;a++) XtWX[a]=0.0;
    for (int i=0;i<n;i++)
        for (int a=0;a<q;a++)
            for (int b=0;b<q;b++)
                XtWX[a*q+b]+=w[i]*Xf[i*q+a]*Xf[i*q+b];
    if (mat_inv(XtWX,q)==0) {
        for (int j=0;j<q;j++) {
            r->se[j]     = sqrt(fabs(XtWX[j*q+j]));
            r->z_stat[j] = (r->se[j]>1e-16) ? r->beta[j]/r->se[j] : 0.0;
            r->p_value[j]= t_pvalue_two_sided(r->z_stat[j], 1e9);
        }
    }

    /* Residual deviance = 2*sum_i [y_i*log(y_i/mu_i) - (y_i - mu_i)]
     * For y_i=0, the y_i*log(y_i/mu_i) term = 0 by convention. */
    double dev = 0.0;
    for (int i=0;i<n;i++) {
        double term1 = (y[i]>0.0) ? y[i]*log(y[i]/mu[i]) : 0.0;
        dev += 2.0*(term1 - (y[i]-mu[i]));
    }
    r->deviance = dev;

    /* AIC = -2*log_likelihood + 2*k
     * log_likelihood = sum_i [y_i*log(mu_i) - mu_i - log(y_i!)]
     * = -dev/2 + sum_i[y_i*log(y_i) - y_i] - sum_i[log(y_i!)]  ... tricky
     * Simpler: compute directly. */
    {
        double llf = 0.0;
        for (int i = 0; i < n; i++) {
            double m_i = (mu[i] < 1e-300) ? 1e-300 : mu[i];
            /* lgamma(y_i+1) = log(y_i!) */
            llf += y[i]*log(m_i) - m_i - lgamma(y[i]+1.0);
        }
        r->aic = -2.0*llf + 2.0*(double)q;
    }

    free(Xf);free(mu);free(eta);free(z);free(w);free(XtWX);free(XtWz);free(beta_new);
    return r;
}

void poisson_free(PoissonResult *r)
{
    if (!r) return;
    free(r->beta); free(r->se); free(r->z_stat); free(r->p_value); free(r);
}

void poisson_predict(const PoissonResult *r, const double *X_new,
                     double *mu_hat, int m)
{
    for (int i=0;i<m;i++) {
        double eta = r->beta[0];
        for (int j=0;j<r->p;j++) eta += r->beta[j+1]*X_new[i*r->p+j];
        mu_hat[i] = safe_exp(eta);
    }
}
