/*
 * multilogreg.h
 * -------------
 * Multinomial logistic regression, ISLP Section 4.3.5.
 *
 * Model (ISLP Eq 4.11):
 *   P(Y=k|X) = exp(beta_k0 + beta_k1*X1 + ... + beta_kp*Xp)
 *              / [1 + sum_{l=1}^{K-1} exp(beta_l0 + ... + beta_lp*Xp)]
 *   for k = 1, ..., K-1
 *
 *   P(Y=K|X) = 1 / [1 + sum_{l=1}^{K-1} exp(...)]    (baseline class K)
 *
 * ISLP Eq 4.12:
 *   log[ P(Y=k|X) / P(Y=K|X) ] = beta_k0 + beta_k1*X1 + ... + beta_kp*Xp
 *
 * Estimation via IRLS (coordinate descent on block diagonal of Hessian):
 *   At each iteration, update beta_k holding all other beta_l fixed,
 *   using the full softmax probabilities (not binary sigmoid) for weights.
 *   This is a Gauss-Seidel approach on the block Newton step.
 *
 * Inference:
 *   Standard errors from diagonal of (X'WX)^{-1} for each class block.
 *   z-statistics and p-values as in logistic regression.
 *
 * The baseline class is the last (numerically largest) class label.
 * Coefficients for the baseline class are all zero by definition.
 */

#ifndef MULTILOGREG_H
#define MULTILOGREG_H

typedef struct {
    int     n;
    int     p;
    int     K;            /* total number of classes                     */
    int    *class_labels; /* (K) distinct labels; last = baseline        */

    /*
     * Coefficient matrices, each row is one non-baseline class.
     * beta_mat: ((K-1) x (p+1)), stored row-major.
     * se_mat, z_mat, p_mat: same shape.
     */
    double *beta_mat;
    double *se_mat;
    double *z_mat;
    double *p_mat;

    double  null_deviance;
    double  deviance;
    double  aic;

    int     converged;
    int     n_iter;
} MultiLogRegResult;

/*
 * Fit multinomial logistic regression.
 *   X       : (n x p) predictor matrix.
 *   y       : (n) integer class labels.
 *   n, p    : dimensions.
 *   K       : number of distinct classes.
 *   max_iter, tol : IRLS convergence.
 */
MultiLogRegResult *multilogreg_fit(const double *X, const int *y,
                                    int n, int p, int K,
                                    int max_iter, double tol);

void multilogreg_free(MultiLogRegResult *r);

/*
 * Posterior probabilities P(Y=k|X_new), shape (n_new x K).
 */
void multilogreg_predict_proba(const MultiLogRegResult *r,
                                const double *X_new, double *proba, int n_new);

/*
 * Predicted class labels (argmax over posterior).
 */
void multilogreg_predict(const MultiLogRegResult *r,
                          const double *X_new, int *labels, int n_new);

#endif /* MULTILOGREG_H */
