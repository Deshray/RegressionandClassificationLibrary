/*
 * classify.c
 * ----------
 * Classification performance metrics: ISLP Chapter 4.
 */

#include "classify.h"
#include <string.h>

/* ------------------------------------------------------------------ */
/* confusion_matrix_binary                                              */
/* ------------------------------------------------------------------ */
BinaryConfusion confusion_matrix_binary(const int *y_true, const int *y_pred,
                                         int n, int positive_class)
{
    BinaryConfusion cm;
    memset(&cm, 0, sizeof(cm));
    cm.n = n;

    for (int i = 0; i < n; i++) {
        int actual = (y_true[i] == positive_class) ? 1 : 0;
        int pred   = (y_pred[i] == positive_class) ? 1 : 0;
        if (actual == 1 && pred == 1) cm.TP++;
        else if (actual == 0 && pred == 0) cm.TN++;
        else if (actual == 0 && pred == 1) cm.FP++;
        else cm.FN++;
    }

    /* Sensitivity = TP / (TP + FN)  — fraction of true 1s correctly caught */
    int pos_total = cm.TP + cm.FN;
    cm.sensitivity = (pos_total > 0) ? (double)cm.TP / (double)pos_total : 0.0;

    /* Specificity = TN / (TN + FP)  — fraction of true 0s correctly caught */
    int neg_total = cm.TN + cm.FP;
    cm.specificity = (neg_total > 0) ? (double)cm.TN / (double)neg_total : 0.0;

    /* Error rate = (FP + FN) / n */
    cm.error_rate = (double)(cm.FP + cm.FN) / (double)n;

    /* Precision = TP / (TP + FP) */
    int pred_pos = cm.TP + cm.FP;
    cm.precision = (pred_pos > 0) ? (double)cm.TP / (double)pred_pos : 0.0;

    return cm;
}

/* ------------------------------------------------------------------ */
/* confusion_matrix  (K x K, row=predicted, col=true)                 */
/* ------------------------------------------------------------------ */
void confusion_matrix(const int *y_true, const int *y_pred, int n,
                      const int *class_labels, int K, int *matrix)
{
    for (int i = 0; i < K * K; i++) matrix[i] = 0;

    for (int i = 0; i < n; i++) {
        int pred_idx = -1, true_idx = -1;
        for (int k = 0; k < K; k++) {
            if (y_pred[i]  == class_labels[k]) pred_idx = k;
            if (y_true[i]  == class_labels[k]) true_idx = k;
        }
        if (pred_idx >= 0 && true_idx >= 0)
            matrix[pred_idx * K + true_idx]++;
    }
}

/* ------------------------------------------------------------------ */
/* accuracy                                                             */
/* ------------------------------------------------------------------ */
double accuracy(const int *y_true, const int *y_pred, int n)
{
    int correct = 0;
    for (int i = 0; i < n; i++)
        if (y_true[i] == y_pred[i]) correct++;
    return (double)correct / (double)n;
}
