/*
 * knn.h
 * -----
 * K-Nearest Neighbors classifier and regressor, ISLP Sections 3.5 and 4.5.2.
 *
 * KNN Classifier (ISLP §4.5.2 / Chapter 2):
 *   Given K and a test point x0, identify the K closest training points N0.
 *   Estimate P(Y=k|X=x0) = (1/K) * sum_{i in N0} I(y_i = k)
 *   Assign class with highest estimated probability.
 *
 * KNN Regressor (ISLP §3.5):
 *   f_hat(x0) = (1/K) * sum_{i in N0} y_i
 *   The average of the K nearest response values.
 *
 * Distance metric: Euclidean (L2).
 *
 * Notes from ISLP:
 *   - KNN classifier dominated by parametric methods when decision boundary
 *     is approximately linear (§4.5.2).
 *   - KNN regression has large bias for small K (overfits) and large variance
 *     for large K (underfits) (§3.5).
 *   - Performance degrades in high dimensions (curse of dimensionality).
 */

#ifndef KNN_H
#define KNN_H

/* ------------------------------------------------------------------ */
/* KNN Classifier                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;            /* number of training observations  */
    int     p;            /* number of features               */
    int     K;            /* number of neighbors              */
    int     n_classes;    /* number of distinct classes       */
    int    *class_labels; /* distinct class labels (n_classes)*/
    double *X_train;      /* (n x p) training data copy       */
    int    *y_train;      /* (n) integer training labels      */
} KNNClassifier;

KNNClassifier *knn_classifier_fit(const double *X, const int *y,
                                   int n, int p, int K);
void           knn_classifier_free(KNNClassifier *m);

/* Predict class labels */
void knn_classifier_predict(const KNNClassifier *m,
                             const double *X_new, int *labels, int n_new);

/* Posterior probabilities shape (n_new x n_classes) */
void knn_classifier_proba(const KNNClassifier *m,
                           const double *X_new, double *proba, int n_new);

/* ------------------------------------------------------------------ */
/* KNN Regressor                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    int     n;
    int     p;
    int     K;
    double *X_train;  /* (n x p) training data copy */
    double *y_train;  /* (n) training response copy */
} KNNRegressor;

KNNRegressor *knn_regressor_fit(const double *X, const double *y,
                                  int n, int p, int K);
void          knn_regressor_free(KNNRegressor *m);

/* Predict responses */
void knn_regressor_predict(const KNNRegressor *m,
                            const double *X_new, double *y_hat, int n_new);

#endif /* KNN_H */
