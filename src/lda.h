/*
 * lda.h
 * -----
 * Generative classifiers from ISLP Chapter 4 Section 4.4:
 *   Linear Discriminant Analysis  (LDA)   — Section 4.4.1 / 4.4.2
 *   Quadratic Discriminant Analysis (QDA) — Section 4.4.3
 *   Naive Bayes (Gaussian)                — Section 4.4.4
 *
 * All three methods use Bayes' theorem (ISLP Eq 4.15):
 *
 *   P(Y=k | X=x) = pi_k * f_k(x) / sum_l [ pi_l * f_l(x) ]
 *
 * They differ in how they model  f_k(x) = Pr(X|Y=k):
 *
 *  LDA: X|Y=k ~ N(mu_k, Sigma)         SHARED covariance
 *       Discriminant function (Eq 4.24):
 *         delta_k(x) = x' Sigma^{-1} mu_k - 0.5*mu_k' Sigma^{-1} mu_k + log(pi_k)
 *       Decision: assign to class k* = argmax_k delta_k(x)
 *
 *  QDA: X|Y=k ~ N(mu_k, Sigma_k)       CLASS-SPECIFIC covariance
 *       Discriminant function (Eq 4.28):
 *         delta_k(x) = -0.5*(x-mu_k)' Sigma_k^{-1} (x-mu_k)
 *                      - 0.5*log|Sigma_k| + log(pi_k)
 *
 *  Naive Bayes: assumes X_1,...,X_p are CONDITIONALLY INDEPENDENT given Y=k.
 *       log posterior ∝ log(pi_k) + sum_j log f_{kj}(x_j)
 *       where f_{kj}(x_j) = N(x_j; mu_kj, sigma_kj^2)  (Gaussian version)
 *
 * Parameter estimation (same for LDA and QDA, ISLP Eq 4.20-4.21):
 *   pi_hat_k  = n_k / n
 *   mu_hat_k  = (1/n_k) * sum_{i: y_i=k} x_i
 *
 * Pooled covariance for LDA (ISLP p.154):
 *   Sigma_hat = (1/(n-K)) * sum_k sum_{i:y_i=k} (x_i - mu_hat_k)(x_i - mu_hat_k)'
 *
 * Class-specific covariance for QDA:
 *   Sigma_hat_k = (1/(n_k-1)) * sum_{i:y_i=k} (x_i-mu_hat_k)(x_i-mu_hat_k)'
 */

#ifndef LDA_H
#define LDA_H

/* ------------------------------------------------------------------ */
/* LDA model structure                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;           /* total training observations                  */
    int     p;           /* number of predictors                         */
    int     K;           /* number of classes                            */
    int    *class_labels;/* distinct class labels (length K)             */

    double *pi;          /* prior probabilities pi_hat_k  (length K)     */
    double *mu;          /* class means, row-major (K x p)               */
    double *Sigma_inv;   /* inverse of pooled covariance (p x p)         */

    /*
     * Pre-computed quantities per class to speed up prediction:
     *   const_k = -0.5 * mu_k' Sigma^{-1} mu_k + log(pi_k)
     * so  delta_k(x) = x' Sigma^{-1} mu_k + const_k
     */
    double *lda_const;   /* length K                                     */

} LDAModel;

/* ------------------------------------------------------------------ */
/* QDA model structure                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;
    int     p;
    int     K;
    int    *class_labels;

    double *pi;          /* prior probs (K)                              */
    double *mu;          /* class means (K x p)                          */
    double *Sigma_inv;   /* class-specific Sigma_k^{-1}, shape (K, p, p) */
    double *log_det;     /* log|Sigma_k| for each class (K)              */

} QDAModel;

/* ------------------------------------------------------------------ */
/* Naive Bayes model structure (Gaussian)                               */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;
    int     p;
    int     K;
    int    *class_labels;

    double *pi;          /* prior probs (K)                              */
    double *mu;          /* class-feature means (K x p)                  */
    double *var;         /* class-feature variances (K x p)              */

} NaiveBayesModel;

/* ------------------------------------------------------------------ */
/* LDA API                                                              */
/* ------------------------------------------------------------------ */

/*
 * Fit LDA.
 *   X      : (n x p) predictor matrix.
 *   y      : (n) integer class labels (arbitrary integers, e.g. 0,1 or 1,2,3).
 *   n, p   : dimensions.
 *   K      : number of distinct classes.
 * Returns NULL on failure.  Caller must call lda_free().
 */
LDAModel *lda_fit(const double *X, const int *y, int n, int p, int K);

void lda_free(LDAModel *m);

/*
 * Predict class labels for new observations.
 *   X_new   : (m x p) new predictors.
 *   labels  : (m) integer output — caller allocates.
 *   m       : number of new observations.
 */
void lda_predict(const LDAModel *m, const double *X_new,
                 int *labels, int n_new);

/*
 * Posterior probabilities for each class.
 *   X_new   : (n_new x p).
 *   post    : (n_new x K) output — caller allocates.
 */
void lda_posterior(const LDAModel *m, const double *X_new,
                   double *post, int n_new);

/* ------------------------------------------------------------------ */
/* QDA API                                                              */
/* ------------------------------------------------------------------ */

QDAModel *qda_fit(const double *X, const int *y, int n, int p, int K);
void      qda_free(QDAModel *m);
void      qda_predict(const QDAModel *m, const double *X_new,
                      int *labels, int n_new);
void      qda_posterior(const QDAModel *m, const double *X_new,
                        double *post, int n_new);

/* ------------------------------------------------------------------ */
/* Naive Bayes API                                                      */
/* ------------------------------------------------------------------ */

NaiveBayesModel *nb_fit(const double *X, const int *y, int n, int p, int K);
void             nb_free(NaiveBayesModel *m);
void             nb_predict(const NaiveBayesModel *m, const double *X_new,
                            int *labels, int n_new);
void             nb_posterior(const NaiveBayesModel *m, const double *X_new,
                              double *post, int n_new);

#endif /* LDA_H */
