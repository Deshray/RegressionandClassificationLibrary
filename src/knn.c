/*
 * knn.c
 * -----
 * K-Nearest Neighbors: ISLP Sections 3.5 and 4.5.2.
 */

#include "knn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------ */
/* Internal: squared Euclidean distance between two p-vectors          */
/* ------------------------------------------------------------------ */
static double sq_dist(const double *a, const double *b, int p)
{
    double s = 0.0;
    for (int j = 0; j < p; j++) { double d = a[j]-b[j]; s += d*d; }
    return s;
}

/* ------------------------------------------------------------------ */
/* Internal: find K nearest neighbor indices in X_train               */
/* Returns an int array of length K (caller must free)                 */
/* ------------------------------------------------------------------ */
static int *knn_find_neighbors(const double *x0, const double *X_train,
                                int n, int p, int K)
{
    /* We keep a rolling heap of the K smallest distances seen so far.
     * Simple O(n*K) approach — sufficient for the scales ISLP uses. */
    double *best_d = (double *)malloc((size_t)K * sizeof(double));
    int    *best_i = (int    *)malloc((size_t)K * sizeof(int));
    if (!best_d || !best_i) { free(best_d); free(best_i); return NULL; }

    for (int k = 0; k < K; k++) { best_d[k] = DBL_MAX; best_i[k] = -1; }

    for (int i = 0; i < n; i++) {
        double d = sq_dist(x0, X_train + i*p, p);
        /* Find the maximum in best_d */
        int worst = 0;
        for (int k = 1; k < K; k++)
            if (best_d[k] > best_d[worst]) worst = k;
        if (d < best_d[worst]) { best_d[worst] = d; best_i[worst] = i; }
    }

    free(best_d);
    return best_i;
}

/* ================================================================== */
/* KNN Classifier                                                       */
/* ================================================================== */

KNNClassifier *knn_classifier_fit(const double *X, const int *y,
                                   int n, int p, int K)
{
    KNNClassifier *m = (KNNClassifier *)calloc(1, sizeof(KNNClassifier));
    if (!m) return NULL;
    m->n = n; m->p = p; m->K = (K > n) ? n : K;

    m->X_train = (double *)malloc((size_t)n * (size_t)p * sizeof(double));
    m->y_train = (int    *)malloc((size_t)n * sizeof(int));
    if (!m->X_train || !m->y_train) { knn_classifier_free(m); return NULL; }

    memcpy(m->X_train, X, (size_t)n*(size_t)p*sizeof(double));
    memcpy(m->y_train, y, (size_t)n*sizeof(int));

    /* Find distinct class labels */
    int *labels_tmp = (int *)calloc((size_t)n, sizeof(int));
    if (!labels_tmp) { knn_classifier_free(m); return NULL; }
    int K_classes = 0;
    for (int i = 0; i < n; i++) {
        int seen = 0;
        for (int k = 0; k < K_classes; k++)
            if (labels_tmp[k] == y[i]) { seen = 1; break; }
        if (!seen) labels_tmp[K_classes++] = y[i];
    }
    /* sort */
    for (int a=0;a<K_classes-1;a++)
        for (int b=a+1;b<K_classes;b++)
            if(labels_tmp[a]>labels_tmp[b]){int t=labels_tmp[a];labels_tmp[a]=labels_tmp[b];labels_tmp[b]=t;}
    m->n_classes   = K_classes;
    m->class_labels= (int *)malloc((size_t)K_classes*sizeof(int));
    if(!m->class_labels){free(labels_tmp);knn_classifier_free(m);return NULL;}
    memcpy(m->class_labels, labels_tmp, (size_t)K_classes*sizeof(int));
    free(labels_tmp);
    return m;
}

void knn_classifier_free(KNNClassifier *m)
{
    if (!m) return;
    free(m->X_train); free(m->y_train); free(m->class_labels); free(m);
}

void knn_classifier_predict(const KNNClassifier *m,
                             const double *X_new, int *labels, int n_new)
{
    int K = m->K, p = m->p;
    int *counts = (int *)malloc((size_t)m->n_classes * sizeof(int));
    if (!counts) return;

    for (int i = 0; i < n_new; i++) {
        int *nbr = knn_find_neighbors(X_new + i*p, m->X_train, m->n, p, K);
        if (!nbr) { labels[i] = m->class_labels[0]; continue; }

        for (int c = 0; c < m->n_classes; c++) counts[c] = 0;
        for (int k = 0; k < K; k++) {
            int yi = m->y_train[nbr[k]];
            for (int c = 0; c < m->n_classes; c++)
                if (m->class_labels[c] == yi) { counts[c]++; break; }
        }
        int best = 0;
        for (int c = 1; c < m->n_classes; c++)
            if (counts[c] > counts[best]) best = c;
        labels[i] = m->class_labels[best];
        free(nbr);
    }
    free(counts);
}

void knn_classifier_proba(const KNNClassifier *m,
                           const double *X_new, double *proba, int n_new)
{
    int K = m->K, p = m->p, C = m->n_classes;
    int *counts = (int *)malloc((size_t)C * sizeof(int));
    if (!counts) return;

    for (int i = 0; i < n_new; i++) {
        int *nbr = knn_find_neighbors(X_new + i*p, m->X_train, m->n, p, K);
        if (!nbr) {
            for (int c = 0; c < C; c++) proba[i*C+c] = 1.0/C;
            continue;
        }
        for (int c = 0; c < C; c++) counts[c] = 0;
        for (int k = 0; k < K; k++) {
            int yi = m->y_train[nbr[k]];
            for (int c = 0; c < C; c++)
                if (m->class_labels[c] == yi) { counts[c]++; break; }
        }
        for (int c = 0; c < C; c++)
            proba[i*C+c] = (double)counts[c] / (double)K;
        free(nbr);
    }
    free(counts);
}

/* ================================================================== */
/* KNN Regressor                                                        */
/* ================================================================== */

KNNRegressor *knn_regressor_fit(const double *X, const double *y,
                                  int n, int p, int K)
{
    KNNRegressor *m = (KNNRegressor *)calloc(1, sizeof(KNNRegressor));
    if (!m) return NULL;
    m->n = n; m->p = p; m->K = (K > n) ? n : K;
    m->X_train = (double *)malloc((size_t)n*(size_t)p*sizeof(double));
    m->y_train = (double *)malloc((size_t)n*sizeof(double));
    if (!m->X_train || !m->y_train) { knn_regressor_free(m); return NULL; }
    memcpy(m->X_train, X, (size_t)n*(size_t)p*sizeof(double));
    memcpy(m->y_train, y, (size_t)n*sizeof(double));
    return m;
}

void knn_regressor_free(KNNRegressor *m)
{
    if (!m) return;
    free(m->X_train); free(m->y_train); free(m);
}

void knn_regressor_predict(const KNNRegressor *m,
                            const double *X_new, double *y_hat, int n_new)
{
    int K = m->K, p = m->p;
    for (int i = 0; i < n_new; i++) {
        int *nbr = knn_find_neighbors(X_new + i*p, m->X_train, m->n, p, K);
        if (!nbr) { y_hat[i] = 0.0; continue; }
        double sum = 0.0;
        for (int k = 0; k < K; k++) sum += m->y_train[nbr[k]];
        y_hat[i] = sum / (double)K;
        free(nbr);
    }
}
