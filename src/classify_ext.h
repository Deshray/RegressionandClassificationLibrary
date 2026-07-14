/*
 * classify_ext.h
 * --------------
 * Extended classification metrics from ISLP Chapter 4.
 *
 * Table 4.6 — Confusion matrix counts
 *   TP, FP, FN, TN (True/False Positive/Negative)
 *
 * Table 4.7 — Derived performance measures:
 *   Sensitivity    = TP/(TP+FN)       (true positive rate, recall, power)
 *   Specificity    = TN/(TN+FP)       (true negative rate)
 *   Precision      = TP/(TP+FP)       (positive predictive value, PPV)
 *   NPV            = TN/(TN+FN)       (negative predictive value)
 *   FDR            = FP/(FP+TP)       (false discovery rate = 1 - precision)
 *   FPR            = FP/(FP+TN)       (false positive rate = 1 - specificity)
 *   F1             = 2*TP/(2*TP+FP+FN) (harmonic mean of precision + recall)
 *   Balanced acc.  = (sensitivity + specificity) / 2
 *   MCC            = (TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
 *
 * ROC Curve (ISLP p.150-151):
 *   Plot of sensitivity (TPR) vs 1-specificity (FPR) as classification
 *   threshold varies over all possible values.
 *   AUC = area under ROC curve via trapezoidal rule.
 *   AUC = 0.5 → no better than chance; AUC = 1.0 → perfect.
 */

#ifndef CLASSIFY_EXT_H
#define CLASSIFY_EXT_H

/* ------------------------------------------------------------------ */
/* Extended binary confusion matrix                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    int    TP, TN, FP, FN, n;

    /* Table 4.7 metrics */
    double sensitivity;        /* recall, TPR, power: TP/(TP+FN)          */
    double specificity;        /* TNR:                TN/(TN+FP)          */
    double precision;          /* PPV:                TP/(TP+FP)          */
    double npv;                /* NPV:                TN/(TN+FN)          */
    double fdr;                /* FDR:                FP/(TP+FP)          */
    double fpr;                /* FPR (1-spec):       FP/(FP+TN)          */
    double f1;                 /* F1:                 harmonic mean       */
    double balanced_accuracy;  /* (sensitivity + specificity) / 2         */
    double mcc;                /* Matthews correlation coefficient         */
    double error_rate;         /* (FP+FN)/n                               */
} BinaryMetrics;

/*
 * Compute full binary metrics.
 *   y_true, y_pred : (n) integer labels.
 *   positive_class : which label counts as positive.
 */
BinaryMetrics binary_metrics(const int *y_true, const int *y_pred,
                              int n, int positive_class);

/* ------------------------------------------------------------------ */
/* ROC curve                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    double *fpr;         /* false positive rate (1-specificity)  (n_points) */
    double *tpr;         /* true positive rate  (sensitivity)    (n_points) */
    double *thresholds;  /* classification thresholds             (n_points) */
    int     n_points;
    double  auc;         /* area under curve (trapezoidal)                   */
} ROCCurve;

/*
 * Compute ROC curve.
 *   y_true : (n) true binary labels (0 or 1).
 *   probs  : (n) predicted probabilities for the positive class.
 *   positive_class : which integer label is "positive" (typically 1).
 * Returns heap-allocated ROCCurve.  Caller must call roc_free().
 */
ROCCurve *roc_curve(const int *y_true, const double *probs,
                    int n, int positive_class);

void roc_free(ROCCurve *r);

/*
 * AUC only (without building the full curve).
 */
double auc_score(const int *y_true, const double *probs,
                 int n, int positive_class);

/* ------------------------------------------------------------------ */
/* Multi-class accuracy                                                 */
/* ------------------------------------------------------------------ */

double accuracy_score(const int *y_true, const int *y_pred, int n);

#endif /* CLASSIFY_EXT_H */
