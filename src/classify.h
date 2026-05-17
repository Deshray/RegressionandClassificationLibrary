/*
 * classify.h / classify.c
 * -----------------------
 * Classification performance metrics from ISLP Chapter 4.
 *
 * Confusion matrix (ISLP Table 4.4 and surrounding discussion):
 *   Rows    = predicted class
 *   Columns = true class
 *
 * For binary classification (classes 0 and 1):
 *   True Negative  (TN): predicted 0, true 0
 *   False Positive (FP): predicted 1, true 0
 *   False Negative (FN): predicted 0, true 1
 *   True Positive  (TP): predicted 1, true 1
 *
 * Derived metrics (ISLP p.153-155):
 *   Sensitivity    = TP / (TP + FN)    — fraction of true positives caught
 *   Specificity    = TN / (TN + FP)    — fraction of true negatives caught
 *   Error rate      = (FP + FN) / n
 *   Precision       = TP / (TP + FP)
 */

#ifndef CLASSIFY_H
#define CLASSIFY_H

/* ------------------------------------------------------------------ */
/* Confusion matrix for binary classifiers                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int TP;   /* true positives   */
    int TN;   /* true negatives   */
    int FP;   /* false positives  */
    int FN;   /* false negatives  */
    int n;    /* total            */

    double sensitivity;  /* TP / (TP + FN)  (ISLP p.153) */
    double specificity;  /* TN / (TN + FP)  (ISLP p.155) */
    double error_rate;   /* (FP + FN) / n                  */
    double precision;    /* TP / (TP + FP)                 */
} BinaryConfusion;

/*
 * Compute confusion matrix and derived metrics for binary classifier.
 *   y_true      : (n) true labels (0 or 1).
 *   y_pred      : (n) predicted labels (0 or 1).
 *   positive_class : which value counts as "positive" (typically 1).
 */
BinaryConfusion confusion_matrix_binary(const int *y_true, const int *y_pred,
                                         int n, int positive_class);

/* ------------------------------------------------------------------ */
/* General K-class confusion matrix                                     */
/* ------------------------------------------------------------------ */

/*
 * Compute K x K confusion matrix.
 *   y_true, y_pred  : (n) integer labels.
 *   class_labels    : (K) distinct class labels in order.
 *   matrix          : (K x K) output, row = predicted, col = true.
 *                     Caller allocates.
 */
void confusion_matrix(const int *y_true, const int *y_pred, int n,
                      const int *class_labels, int K, int *matrix);

/*
 * Overall accuracy (proportion correctly classified).
 */
double accuracy(const int *y_true, const int *y_pred, int n);

#endif /* CLASSIFY_H */
