/*
 * multilogreg.c
 * -------------
 * Multinomial logistic regression: ISLP Section 4.3.5.
 *
 * Algorithm: Block-diagonal IRLS (Gauss-Seidel Newton).
 * At each iteration:
 *   1. Compute softmax probabilities pi_ik for all i, k.
 *   2. For each non-baseline class k (k=0,...,K-2):
 *      a. Weights w_{ki} = pi_ik * (1 - pi_ik)  (same as binary logistic)
 *      b. Working response z_{ki} = eta_{ki} + (y_{ki} - pi_ik)/w_{ki}
 *      c. Update beta_k = (X'W_kX)^{-1} X'W_kz_k  via WLS
 *   3. Check convergence on max change in any beta.
 *
 * This correctly uses the full softmax at each step, unlike K independent
 * binary regressions which would not.
 */

#include "multilogreg.h"
#include "linalg.h"
#include "stat_dist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------ */
/* Softmax: compute K probabilities from K-1 linear predictors + 0    */
/* eta: (K-1) vector for classes 0..K-2; baseline (K-1) has eta=0    */
/* out: (K) probabilities, sums to 1                                   */
/* ------------------------------------------------------------------ */
static void softmax(const double *eta, int K, double *out)
{
    /* out[k] = exp(eta[k]) / sum, with eta[K-1] = 0 */
    double max_eta = 0.0;
    for (int k = 0; k < K-1; k++) if (eta[k] > max_eta) max_eta = eta[k];

    double denom = exp(0.0 - max_eta);   /* baseline: eta = 0 */
    for (int k = 0; k < K-1; k++) {
        out[k] = exp(eta[k] - max_eta);
        denom += out[k];
    }
    for (int k = 0; k < K-1; k++) out[k] /= denom;
    out[K-1] = exp(0.0 - max_eta) / denom;
}

/* ------------------------------------------------------------------ */
/* Multinomial log-likelihood                                           */
/* ------------------------------------------------------------------ */
static double multi_loglik(const double *prob, const int *y_int,
                            const int *class_labels, int n, int K)
{
    double ll = 0.0;
    const double EPS = 1e-300;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < K; k++) {
            if (y_int[i] == class_labels[k]) {
                double p = (prob[i*K+k] < EPS) ? EPS : prob[i*K+k];
                ll += log(p);
                break;
            }
        }
    }
    return ll;
}

/* ------------------------------------------------------------------ */
/* multilogreg_fit                                                      */
/* ------------------------------------------------------------------ */
MultiLogRegResult *multilogreg_fit(const double *X, const int *y,
                                    int n, int p, int K,
                                    int max_iter, double tol)
{
    int q  = p + 1;       /* full design width */
    int Km1 = K - 1;      /* number of non-baseline classes */

    MultiLogRegResult *r = (MultiLogRegResult *)calloc(1, sizeof(MultiLogRegResult));
    if (!r) return NULL;
    r->n = n; r->p = p; r->K = K;

    r->class_labels = (int    *)malloc((size_t)K * sizeof(int));
    r->beta_mat     = (double *)calloc((size_t)Km1 * (size_t)q, sizeof(double));
    r->se_mat       = (double *)calloc((size_t)Km1 * (size_t)q, sizeof(double));
    r->z_mat        = (double *)calloc((size_t)Km1 * (size_t)q, sizeof(double));
    r->p_mat        = (double *)calloc((size_t)Km1 * (size_t)q, sizeof(double));
    if (!r->class_labels||!r->beta_mat||!r->se_mat||!r->z_mat||!r->p_mat) {
        multilogreg_free(r); return NULL;
    }

    /* Detect and sort class labels; last = baseline */
    {
        int found=0;
        for (int i=0;i<n&&found<K;i++) {
            int seen=0;
            for (int k=0;k<found;k++) if(r->class_labels[k]==y[i]){seen=1;break;}
            if (!seen) r->class_labels[found++]=y[i];
        }
        for (int a=0;a<K-1;a++) for(int b=a+1;b<K;b++)
            if(r->class_labels[a]>r->class_labels[b]) {
                int t=r->class_labels[a]; r->class_labels[a]=r->class_labels[b]; r->class_labels[b]=t;
            }
    }

    /* Build full design matrix Xf = [1|X] (n x q) */
    double *Xf = (double *)malloc((size_t)n*(size_t)q*sizeof(double));
    if (!Xf) { multilogreg_free(r); return NULL; }
    for (int i=0;i<n;i++) {
        Xf[i*q]=1.0;
        for (int j=0;j<p;j++) Xf[i*q+j+1]=X[i*p+j];
    }

    /* Working arrays */
    double *prob = (double *)calloc((size_t)n*(size_t)K, sizeof(double));  /* posterior (n x K) */
    double *eta  = (double *)calloc((size_t)n*(size_t)Km1, sizeof(double));/* linear pred (n x Km1) */
    double *XtWX = (double *)malloc((size_t)q*(size_t)q*sizeof(double));
    double *XtWz = (double *)malloc((size_t)q*sizeof(double));
    double *beta_k_new = (double *)malloc((size_t)q*sizeof(double));
    if (!prob||!eta||!XtWX||!XtWz||!beta_k_new) {
        free(Xf);free(prob);free(eta);free(XtWX);free(XtWz);free(beta_k_new);
        multilogreg_free(r); return NULL;
    }

    /* Null deviance: intercept-only model */
    /* Prior probs: pi_k = n_k/n */
    double null_ll = 0.0;
    for (int i=0;i<n;i++) {
        for (int k=0;k<K;k++) {
            if (y[i]==r->class_labels[k]) {
                /* count class k */
                int nk=0; for(int ii=0;ii<n;ii++) if(y[ii]==r->class_labels[k]) nk++;
                double pi = (double)nk/(double)n;
                null_ll += log((pi<1e-300)?1e-300:pi);
                break;
            }
        }
    }
    r->null_deviance = -2.0*null_ll;

    /* IRLS */
    r->converged = 0;
    for (int iter=0; iter<max_iter; iter++) {
        r->n_iter = iter+1;

        /* Compute eta[i][k] = Xf[i] . beta_mat[k] for k=0..Km1-1 */
        for (int i=0;i<n;i++) {
            double *eta_row = eta + i*Km1;
            for (int k=0;k<Km1;k++) {
                double *bk = r->beta_mat + k*q;
                eta_row[k] = 0.0;
                for (int j=0;j<q;j++) eta_row[k] += Xf[i*q+j]*bk[j];
            }
        }

        /* Compute softmax probabilities */
        for (int i=0;i<n;i++)
            softmax(eta+i*Km1, K, prob+i*K);

        /* Update each non-baseline class k using block WLS */
        double max_delta = 0.0;
        for (int k=0;k<Km1;k++) {
            double *bk = r->beta_mat + k*q;

            /* XtWX and XtWz for class k */
            for (int a=0;a<q*q;a++) XtWX[a]=0.0;
            for (int a=0;a<q;a++) XtWz[a]=0.0;

            for (int i=0;i<n;i++) {
                double pik = prob[i*K+k];
                double wik = pik*(1.0-pik);
                if (wik<1e-15) wik=1e-15;

                /* y_{ik} = 1 if class k, else 0 */
                int yik = (y[i]==r->class_labels[k]) ? 1 : 0;

                double zik = eta[i*Km1+k] + ((double)yik - pik)/wik;

                double wz = wik*zik;
                for (int a=0;a<q;a++) {
                    XtWz[a] += wz*Xf[i*q+a];
                    for (int b=0;b<q;b++)
                        XtWX[a*q+b] += wik*Xf[i*q+a]*Xf[i*q+b];
                }
            }

            double *XtWX_copy = (double *)malloc((size_t)q*(size_t)q*sizeof(double));
            if (!XtWX_copy) goto cleanup;
            memcpy(XtWX_copy, XtWX, (size_t)q*(size_t)q*sizeof(double));
            if (mat_inv(XtWX_copy,q)!=0) { free(XtWX_copy); continue; }
            for (int a=0;a<q;a++) {
                beta_k_new[a]=0.0;
                for (int b=0;b<q;b++) beta_k_new[a]+=XtWX_copy[a*q+b]*XtWz[b];
            }
            free(XtWX_copy);

            for (int j=0;j<q;j++) {
                double d=fabs(beta_k_new[j]-bk[j]); if(d>max_delta) max_delta=d;
                bk[j]=beta_k_new[j];
            }
        }

        if (max_delta < tol) { r->converged=1; break; }
    }

    /* SE, z-stats, p-values: recompute final probs and information matrices */
    for (int i=0;i<n;i++) {
        double *eta_row = eta+i*Km1;
        for (int k=0;k<Km1;k++) {
            double *bk = r->beta_mat+k*q;
            eta_row[k]=0.0;
            for (int j=0;j<q;j++) eta_row[k]+=Xf[i*q+j]*bk[j];
        }
        softmax(eta+i*Km1, K, prob+i*K);
    }
    for (int k=0;k<Km1;k++) {
        for (int a=0;a<q*q;a++) XtWX[a]=0.0;
        for (int i=0;i<n;i++) {
            double pik=prob[i*K+k];
            double wik=pik*(1.0-pik);
            if(wik<1e-15) wik=1e-15;
            for (int a=0;a<q;a++) for (int b=0;b<q;b++)
                XtWX[a*q+b]+=wik*Xf[i*q+a]*Xf[i*q+b];
        }
        if (mat_inv(XtWX,q)==0) {
            double *bk=r->beta_mat+k*q;
            double *sk=r->se_mat+k*q;
            double *zk=r->z_mat+k*q;
            double *pk=r->p_mat+k*q;
            for (int j=0;j<q;j++) {
                sk[j]=sqrt(fabs(XtWX[j*q+j]));
                zk[j]=(sk[j]>1e-16)?bk[j]/sk[j]:0.0;
                pk[j]=t_pvalue_two_sided(zk[j],1e9);
            }
        }
    }

    /* Deviance */
    double ll = multi_loglik(prob, y, r->class_labels, n, K);
    r->deviance = -2.0*ll;
    r->aic      = r->deviance + 2.0*(double)(Km1*q);

cleanup:
    free(Xf);free(prob);free(eta);free(XtWX);free(XtWz);free(beta_k_new);
    return r;
}

void multilogreg_free(MultiLogRegResult *r)
{
    if (!r) return;
    free(r->class_labels); free(r->beta_mat); free(r->se_mat);
    free(r->z_mat); free(r->p_mat); free(r);
}

void multilogreg_predict_proba(const MultiLogRegResult *r,
                                const double *X_new, double *proba, int n_new)
{
    int p=r->p, K=r->K, q=p+1, Km1=K-1;
    double *eta = (double *)malloc((size_t)Km1*sizeof(double));
    if (!eta) return;

    for (int i=0;i<n_new;i++) {
        for (int k=0;k<Km1;k++) {
            double *bk=r->beta_mat+k*q;
            eta[k]=bk[0];
            for (int j=0;j<p;j++) eta[k]+=bk[j+1]*X_new[i*p+j];
        }
        softmax(eta, K, proba+i*K);
    }
    free(eta);
}

void multilogreg_predict(const MultiLogRegResult *r,
                          const double *X_new, int *labels, int n_new)
{
    int K=r->K;
    double *proba=(double *)malloc((size_t)n_new*(size_t)K*sizeof(double));
    if (!proba) return;
    multilogreg_predict_proba(r, X_new, proba, n_new);
    for (int i=0;i<n_new;i++) {
        int best=0;
        for (int k=1;k<K;k++) if(proba[i*K+k]>proba[i*K+best]) best=k;
        labels[i]=r->class_labels[best];
    }
    free(proba);
}
