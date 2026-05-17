/*
 * lda.c
 * -----
 * LDA, QDA, and Gaussian Naive Bayes — ISLP Chapter 4, Section 4.4.
 *
 * Every formula references the relevant ISLP equation number.
 */

#include "lda.h"
#include "linalg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ================================================================== */
/* Shared internal utilities                                            */
/* ================================================================== */

/*
 * Count observations per class and find class indices.
 * `y`            : (n) integer labels.
 * `class_labels` : (K) distinct labels (already sorted/provided).
 * `n_k`          : (K) output — count of observations in each class.
 */
static void count_classes(const int *y, int n, const int *class_labels,
                           int K, int *n_k)
{
    for (int k = 0; k < K; k++) n_k[k] = 0;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < K; k++) {
            if (y[i] == class_labels[k]) { n_k[k]++; break; }
        }
    }
}

/*
 * Compute class-specific mean vectors.
 * `mu`  : (K x p) output, row-major.
 */
static void compute_class_means(const double *X, const int *y, int n, int p,
                                 const int *class_labels, const int *n_k,
                                 int K, double *mu)
{
    /* zero out */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < p; j++)
            mu[k*p + j] = 0.0;

    /* accumulate */
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < K; k++) {
            if (y[i] == class_labels[k]) {
                for (int j = 0; j < p; j++)
                    mu[k*p + j] += X[i*p + j];
                break;
            }
        }
    }

    /* divide by n_k */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < p; j++)
            mu[k*p + j] /= (double)n_k[k];
}

/* ================================================================== */
/* LDA                                                                  */
/* ================================================================== */

LDAModel *lda_fit(const double *X, const int *y, int n, int p, int K)
{
    LDAModel *m = (LDAModel *)calloc(1, sizeof(LDAModel));
    if (!m) return NULL;
    m->n = n; m->p = p; m->K = K;

    m->class_labels = (int    *)malloc((size_t)K * sizeof(int));
    m->pi           = (double *)malloc((size_t)K * sizeof(double));
    m->mu           = (double *)malloc((size_t)K * (size_t)p * sizeof(double));
    m->Sigma_inv    = (double *)malloc((size_t)p * (size_t)p * sizeof(double));
    m->lda_const    = (double *)malloc((size_t)K * sizeof(double));
    if (!m->class_labels || !m->pi || !m->mu || !m->Sigma_inv || !m->lda_const) {
        lda_free(m); return NULL;
    }

    /* ---- Detect unique class labels (assume caller passes sorted labels) ---- */
    /* Simple: caller provides K distinct labels in class_labels array.
     * Here we auto-detect the K smallest distinct values. */
    {
        /* Collect unique labels */
        int found = 0;
        for (int i = 0; i < n && found < K; i++) {
            int seen = 0;
            for (int k = 0; k < found; k++)
                if (m->class_labels[k] == y[i]) { seen = 1; break; }
            if (!seen) m->class_labels[found++] = y[i];
        }
        /* simple sort */
        for (int a = 0; a < K-1; a++)
            for (int b = a+1; b < K; b++)
                if (m->class_labels[a] > m->class_labels[b]) {
                    int tmp = m->class_labels[a];
                    m->class_labels[a] = m->class_labels[b];
                    m->class_labels[b] = tmp;
                }
    }

    int *n_k = (int *)calloc((size_t)K, sizeof(int));
    if (!n_k) { lda_free(m); return NULL; }
    count_classes(y, n, m->class_labels, K, n_k);

    /* ---- pi_hat_k = n_k / n  (ISLP Eq 4.21) ---- */
    for (int k = 0; k < K; k++)
        m->pi[k] = (double)n_k[k] / (double)n;

    /* ---- mu_hat_k  (ISLP Eq 4.20) ---- */
    compute_class_means(X, y, n, p, m->class_labels, n_k, K, m->mu);

    /* ---- Pooled within-class covariance  Sigma_hat  (ISLP p.154)
     *
     *   Sigma_hat = (1/(n-K)) * sum_k sum_{i:y_i=k} (x_i - mu_k)(x_i - mu_k)'
     *
     * Divisor is (n-K), the unbiased pooled estimator.
     */
    for (int a = 0; a < p * p; a++) m->Sigma_inv[a] = 0.0;

    for (int i = 0; i < n; i++) {
        /* Find class of observation i */
        double *mu_k = NULL;
        for (int k = 0; k < K; k++) {
            if (y[i] == m->class_labels[k]) { mu_k = m->mu + k*p; break; }
        }
        if (!mu_k) continue;

        /* Outer product (x_i - mu_k)(x_i - mu_k)' */
        for (int a = 0; a < p; a++) {
            double da = X[i*p + a] - mu_k[a];
            for (int b = 0; b < p; b++) {
                double db = X[i*p + b] - mu_k[b];
                m->Sigma_inv[a*p + b] += da * db;
            }
        }
    }

    double scale = 1.0 / (double)(n - K);
    for (int a = 0; a < p * p; a++) m->Sigma_inv[a] *= scale;

    /* Invert to get Sigma^{-1} (stored in-place) */
    if (mat_inv(m->Sigma_inv, p) != 0) {
        free(n_k); lda_free(m); return NULL;
    }

    /* ---- Pre-compute lda_const_k = -0.5 * mu_k' Sigma^{-1} mu_k + log(pi_k)
     *
     * This lets prediction evaluate  delta_k(x) = x' Sigma^{-1} mu_k + lda_const_k
     * which is O(p) per class rather than O(p^2).
     */
    double *tmp = (double *)malloc((size_t)p * sizeof(double));
    if (!tmp) { free(n_k); lda_free(m); return NULL; }

    for (int k = 0; k < K; k++) {
        double *mu_k = m->mu + k * p;
        /* tmp = Sigma^{-1} mu_k */
        mat_vec_mul(m->Sigma_inv, mu_k, tmp, p, p);
        /* mu_k' Sigma^{-1} mu_k */
        double quad = vec_dot(mu_k, tmp, p);
        m->lda_const[k] = -0.5 * quad + log(m->pi[k]);
    }

    free(tmp); free(n_k);
    return m;
}

void lda_free(LDAModel *m)
{
    if (!m) return;
    free(m->class_labels); free(m->pi); free(m->mu);
    free(m->Sigma_inv); free(m->lda_const);
    free(m);
}

/*
 * Compute discriminant scores  delta_k(x) for a single observation x.
 * Caller provides `scores` (length K).
 *
 * ISLP Eq 4.24:  delta_k(x) = x' Sigma^{-1} mu_k - 0.5 mu_k' Sigma^{-1} mu_k + log(pi_k)
 *               = x' Sigma^{-1} mu_k + lda_const_k
 */
static void lda_scores(const LDAModel *m, const double *x, double *scores)
{
    int p = m->p, K = m->K;
    /* SigInv_x = Sigma^{-1} @ x  (p) */
    double *SigInv_x = (double *)malloc((size_t)p * sizeof(double));
    if (!SigInv_x) {
        for (int k = 0; k < K; k++) scores[k] = -DBL_MAX;
        return;
    }
    mat_vec_mul(m->Sigma_inv, x, SigInv_x, p, p);

    for (int k = 0; k < K; k++) {
        /* x' Sigma^{-1} mu_k = SigInv_x . mu_k */
        scores[k] = vec_dot(SigInv_x, m->mu + k*p, p) + m->lda_const[k];
    }
    free(SigInv_x);
}

void lda_predict(const LDAModel *m, const double *X_new,
                 int *labels, int n_new)
{
    int K = m->K;
    double *scores = (double *)malloc((size_t)K * sizeof(double));
    if (!scores) return;

    for (int i = 0; i < n_new; i++) {
        lda_scores(m, X_new + i * m->p, scores);
        int best = 0;
        for (int k = 1; k < K; k++)
            if (scores[k] > scores[best]) best = k;
        labels[i] = m->class_labels[best];
    }
    free(scores);
}

void lda_posterior(const LDAModel *m, const double *X_new,
                   double *post, int n_new)
{
    int K = m->K;
    double *scores = (double *)malloc((size_t)K * sizeof(double));
    if (!scores) return;

    for (int i = 0; i < n_new; i++) {
        lda_scores(m, X_new + i * m->p, scores);
        /* Convert log-scores to probabilities via softmax */
        double max_s = scores[0];
        for (int k = 1; k < K; k++)
            if (scores[k] > max_s) max_s = scores[k];
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            post[i*K + k] = exp(scores[k] - max_s);
            sum += post[i*K + k];
        }
        for (int k = 0; k < K; k++) post[i*K + k] /= sum;
    }
    free(scores);
}

/* ================================================================== */
/* QDA                                                                  */
/* ================================================================== */

QDAModel *qda_fit(const double *X, const int *y, int n, int p, int K)
{
    QDAModel *m = (QDAModel *)calloc(1, sizeof(QDAModel));
    if (!m) return NULL;
    m->n = n; m->p = p; m->K = K;

    m->class_labels = (int    *)malloc((size_t)K * sizeof(int));
    m->pi           = (double *)malloc((size_t)K * sizeof(double));
    m->mu           = (double *)malloc((size_t)K * (size_t)p * sizeof(double));
    m->Sigma_inv    = (double *)malloc((size_t)K * (size_t)p * (size_t)p * sizeof(double));
    m->log_det      = (double *)malloc((size_t)K * sizeof(double));
    if (!m->class_labels||!m->pi||!m->mu||!m->Sigma_inv||!m->log_det) {
        qda_free(m); return NULL;
    }

    /* Auto-detect class labels (same logic as LDA) */
    {
        int found = 0;
        for (int i = 0; i < n && found < K; i++) {
            int seen = 0;
            for (int k = 0; k < found; k++)
                if (m->class_labels[k] == y[i]) { seen=1; break; }
            if (!seen) m->class_labels[found++] = y[i];
        }
        for (int a=0;a<K-1;a++)
            for (int b=a+1;b<K;b++)
                if (m->class_labels[a]>m->class_labels[b]) {
                    int t=m->class_labels[a];
                    m->class_labels[a]=m->class_labels[b];
                    m->class_labels[b]=t;
                }
    }

    int *n_k = (int *)calloc((size_t)K, sizeof(int));
    if (!n_k) { qda_free(m); return NULL; }
    count_classes(y, n, m->class_labels, K, n_k);

    /* pi_hat_k */
    for (int k = 0; k < K; k++)
        m->pi[k] = (double)n_k[k] / (double)n;

    /* mu_hat_k */
    compute_class_means(X, y, n, p, m->class_labels, n_k, K, m->mu);

    /* ---- Class-specific covariance  Sigma_hat_k  (ISLP 4.4.3)
     *
     *   Sigma_hat_k = (1/(n_k-1)) * sum_{i:y_i=k} (x_i-mu_k)(x_i-mu_k)'
     */
    for (int k = 0; k < K; k++) {
        double *Sk = m->Sigma_inv + k * p * p;
        double *mu_k = m->mu + k * p;

        /* Accumulate scatter matrix */
        for (int a = 0; a < p*p; a++) Sk[a] = 0.0;

        for (int i = 0; i < n; i++) {
            if (y[i] != m->class_labels[k]) continue;
            for (int a = 0; a < p; a++) {
                double da = X[i*p+a] - mu_k[a];
                for (int b = 0; b < p; b++) {
                    double db = X[i*p+b] - mu_k[b];
                    Sk[a*p+b] += da * db;
                }
            }
        }

        double denom = (n_k[k] > 1) ? (double)(n_k[k]-1) : 1.0;
        for (int a = 0; a < p*p; a++) Sk[a] /= denom;

        /* Invert Sk in-place, compute log|Sk| */
        double ld;
        if (mat_inv_logdet(Sk, p, &ld) != 0) {
            /* singular class covariance — add tiny regularization */
            for (int a = 0; a < p; a++) Sk[a*p+a] += 1e-6;
            mat_inv_logdet(Sk, p, &ld);
        }
        m->log_det[k] = ld;
    }

    free(n_k);
    return m;
}

void qda_free(QDAModel *m)
{
    if (!m) return;
    free(m->class_labels); free(m->pi); free(m->mu);
    free(m->Sigma_inv); free(m->log_det);
    free(m);
}

/*
 * QDA discriminant score for a single observation x and class k.
 * ISLP Eq 4.28:
 *   delta_k(x) = -0.5*(x-mu_k)' Sigma_k^{-1} (x-mu_k) - 0.5*log|Sigma_k| + log(pi_k)
 */
static void qda_scores(const QDAModel *m, const double *x, double *scores)
{
    int p = m->p, K = m->K;
    double *diff = (double *)malloc((size_t)p * sizeof(double));
    double *tmp  = (double *)malloc((size_t)p * sizeof(double));
    if (!diff || !tmp) {
        for (int k=0;k<K;k++) scores[k]=-DBL_MAX;
        free(diff); free(tmp); return;
    }

    for (int k = 0; k < K; k++) {
        double *mu_k    = m->mu + k*p;
        double *SigInvk = m->Sigma_inv + k*p*p;

        /* diff = x - mu_k */
        for (int j = 0; j < p; j++) diff[j] = x[j] - mu_k[j];

        /* tmp = Sigma_k^{-1} @ diff */
        mat_vec_mul(SigInvk, diff, tmp, p, p);

        /* quadratic form  diff' Sigma_k^{-1} diff */
        double quad = vec_dot(diff, tmp, p);

        scores[k] = -0.5*quad - 0.5*m->log_det[k] + log(m->pi[k]);
    }
    free(diff); free(tmp);
}

void qda_predict(const QDAModel *m, const double *X_new,
                 int *labels, int n_new)
{
    int K = m->K;
    double *scores = (double *)malloc((size_t)K * sizeof(double));
    if (!scores) return;

    for (int i = 0; i < n_new; i++) {
        qda_scores(m, X_new + i*m->p, scores);
        int best = 0;
        for (int k=1;k<K;k++) if (scores[k]>scores[best]) best=k;
        labels[i] = m->class_labels[best];
    }
    free(scores);
}

void qda_posterior(const QDAModel *m, const double *X_new,
                   double *post, int n_new)
{
    int K = m->K;
    double *scores = (double *)malloc((size_t)K * sizeof(double));
    if (!scores) return;

    for (int i = 0; i < n_new; i++) {
        qda_scores(m, X_new + i*m->p, scores);
        double max_s = scores[0];
        for (int k=1;k<K;k++) if(scores[k]>max_s) max_s=scores[k];
        double sum = 0.0;
        for (int k=0;k<K;k++) { post[i*K+k]=exp(scores[k]-max_s); sum+=post[i*K+k]; }
        for (int k=0;k<K;k++) post[i*K+k]/=sum;
    }
    free(scores);
}

/* ================================================================== */
/* Naive Bayes (Gaussian)                                               */
/* ================================================================== */

NaiveBayesModel *nb_fit(const double *X, const int *y, int n, int p, int K)
{
    NaiveBayesModel *m = (NaiveBayesModel *)calloc(1, sizeof(NaiveBayesModel));
    if (!m) return NULL;
    m->n=n; m->p=p; m->K=K;

    m->class_labels = (int    *)malloc((size_t)K * sizeof(int));
    m->pi           = (double *)malloc((size_t)K * sizeof(double));
    m->mu           = (double *)malloc((size_t)K * (size_t)p * sizeof(double));
    m->var          = (double *)malloc((size_t)K * (size_t)p * sizeof(double));
    if (!m->class_labels||!m->pi||!m->mu||!m->var) { nb_free(m); return NULL; }

    /* Auto-detect class labels */
    {
        int found = 0;
        for (int i = 0; i < n && found < K; i++) {
            int seen = 0;
            for (int k=0;k<found;k++) if(m->class_labels[k]==y[i]){seen=1;break;}
            if (!seen) m->class_labels[found++]=y[i];
        }
        for (int a=0;a<K-1;a++)
            for (int b=a+1;b<K;b++)
                if(m->class_labels[a]>m->class_labels[b]){
                    int t=m->class_labels[a];
                    m->class_labels[a]=m->class_labels[b];
                    m->class_labels[b]=t;
                }
    }

    int *n_k = (int *)calloc((size_t)K, sizeof(int));
    if (!n_k) { nb_free(m); return NULL; }
    count_classes(y, n, m->class_labels, K, n_k);

    for (int k=0;k<K;k++) m->pi[k]=(double)n_k[k]/(double)n;

    compute_class_means(X, y, n, p, m->class_labels, n_k, K, m->mu);

    /* Class-feature variances (ISLP 4.4.4):
     *   sigma_hat_kj^2 = (1/(n_k-1)) * sum_{i:y_i=k} (x_ij - mu_kj)^2
     *
     * Naive Bayes assumes conditional independence of features given class,
     * so we estimate a separate variance for each (class, feature) pair.
     */
    for (int k=0;k<K;k++)
        for (int j=0;j<p;j++)
            m->var[k*p+j] = 0.0;

    for (int i=0;i<n;i++) {
        for (int k=0;k<K;k++) {
            if (y[i]!=m->class_labels[k]) continue;
            for (int j=0;j<p;j++) {
                double d = X[i*p+j] - m->mu[k*p+j];
                m->var[k*p+j] += d*d;
            }
            break;
        }
    }

    for (int k=0;k<K;k++) {
        double denom = (n_k[k]>1) ? (double)(n_k[k]-1) : 1.0;
        for (int j=0;j<p;j++) {
            m->var[k*p+j] /= denom;
            if (m->var[k*p+j] < 1e-15) m->var[k*p+j] = 1e-15; /* guard */
        }
    }

    free(n_k);
    return m;
}

void nb_free(NaiveBayesModel *m)
{
    if (!m) return;
    free(m->class_labels); free(m->pi); free(m->mu); free(m->var);
    free(m);
}

/*
 * Log posterior (unnormalised) for Naive Bayes:
 *
 *   log P(Y=k | x) ∝ log(pi_k) + sum_j log N(x_j; mu_kj, sigma_kj^2)
 *
 * where  log N(x; mu, sigma^2) = -0.5*log(2*pi) - 0.5*log(sigma^2) - (x-mu)^2/(2*sigma^2)
 *
 * The -0.5*log(2*pi) term is constant across classes and cancels in the argmax.
 */
static void nb_log_posterior(const NaiveBayesModel *m, const double *x, double *lp)
{
    int p=m->p, K=m->K;
    const double LOG_2PI = 1.8378770664093455; /* log(2*pi) */

    for (int k=0;k<K;k++) {
        double val = log(m->pi[k]);
        for (int j=0;j<p;j++) {
            double mu_kj  = m->mu[k*p+j];
            double var_kj = m->var[k*p+j];
            double d      = x[j] - mu_kj;
            val += -0.5*LOG_2PI - 0.5*log(var_kj) - 0.5*d*d/var_kj;
        }
        lp[k] = val;
    }
}

void nb_predict(const NaiveBayesModel *m, const double *X_new,
                int *labels, int n_new)
{
    int K = m->K;
    double *lp = (double *)malloc((size_t)K * sizeof(double));
    if (!lp) return;

    for (int i=0;i<n_new;i++) {
        nb_log_posterior(m, X_new+i*m->p, lp);
        int best=0;
        for (int k=1;k<K;k++) if(lp[k]>lp[best]) best=k;
        labels[i] = m->class_labels[best];
    }
    free(lp);
}

void nb_posterior(const NaiveBayesModel *m, const double *X_new,
                  double *post, int n_new)
{
    int K = m->K;
    double *lp = (double *)malloc((size_t)K * sizeof(double));
    if (!lp) return;

    for (int i=0;i<n_new;i++) {
        nb_log_posterior(m, X_new+i*m->p, lp);
        double max_lp = lp[0];
        for (int k=1;k<K;k++) if(lp[k]>max_lp) max_lp=lp[k];
        double sum=0.0;
        for (int k=0;k<K;k++) { post[i*K+k]=exp(lp[k]-max_lp); sum+=post[i*K+k]; }
        for (int k=0;k<K;k++) post[i*K+k]/=sum;
    }
    free(lp);
}
