/*
 * classify_ext.c
 * --------------
 * Extended classification metrics: ISLP Chapter 4, Tables 4.6 and 4.7.
 */

#include "classify_ext.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================== */
/* binary_metrics                                                       */
/* ================================================================== */

BinaryMetrics binary_metrics(const int *y_true, const int *y_pred,
                              int n, int positive_class)
{
    BinaryMetrics m;
    memset(&m, 0, sizeof(m));
    m.n = n;

    for (int i = 0; i < n; i++) {
        int act = (y_true[i] == positive_class) ? 1 : 0;
        int prd = (y_pred[i] == positive_class) ? 1 : 0;
        if      (act == 1 && prd == 1) m.TP++;
        else if (act == 0 && prd == 0) m.TN++;
        else if (act == 0 && prd == 1) m.FP++;
        else                           m.FN++;
    }

    int pos_total = m.TP + m.FN;
    int neg_total = m.TN + m.FP;
    int pred_pos  = m.TP + m.FP;
    int pred_neg  = m.TN + m.FN;

    m.sensitivity        = pos_total  ? (double)m.TP / pos_total  : 0.0;
    m.specificity        = neg_total  ? (double)m.TN / neg_total  : 0.0;
    m.precision          = pred_pos   ? (double)m.TP / pred_pos   : 0.0;
    m.npv                = pred_neg   ? (double)m.TN / pred_neg   : 0.0;
    m.fdr                = pred_pos   ? (double)m.FP / pred_pos   : 0.0;
    m.fpr                = neg_total  ? (double)m.FP / neg_total  : 0.0;
    m.error_rate         = n          ? (double)(m.FP + m.FN) / n : 0.0;
    m.balanced_accuracy  = 0.5 * (m.sensitivity + m.specificity);

    /* F1 = 2*TP / (2*TP + FP + FN) = 2*precision*recall/(precision+recall) */
    int denom_f1 = 2 * m.TP + m.FP + m.FN;
    m.f1 = denom_f1 ? 2.0 * m.TP / denom_f1 : 0.0;

    /* Matthews Correlation Coefficient */
    double mcc_denom = sqrt(
        (double)(m.TP + m.FP) * (m.TP + m.FN) *
        (m.TN + m.FP) * (m.TN + m.FN));
    m.mcc = (mcc_denom > 1e-15)
        ? ((double)(m.TP * m.TN) - (double)(m.FP * m.FN)) / mcc_denom
        : 0.0;

    return m;
}

/* ================================================================== */
/* ROC curve                                                            */
/* ================================================================== */

/* Sort helper: sort indices by descending probability */
typedef struct { double prob; int label; } _ProbLabel;

static int _cmp_desc(const void *a, const void *b)
{
    double pa = ((_ProbLabel*)a)->prob;
    double pb = ((_ProbLabel*)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return  1;
    return 0;
}

ROCCurve *roc_curve(const int *y_true, const double *probs,
                    int n, int positive_class)
{
    /* We build the ROC curve by sweeping thresholds from high to low prob.
     * At each unique threshold, compute (FPR, TPR).
     */
    _ProbLabel *sorted = (_ProbLabel *)malloc((size_t)n * sizeof(_ProbLabel));
    if (!sorted) return NULL;

    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        sorted[i].prob  = probs[i];
        sorted[i].label = (y_true[i] == positive_class) ? 1 : 0;
        if (sorted[i].label == 1) n_pos++;
        else                      n_neg++;
    }
    qsort(sorted, (size_t)n, sizeof(_ProbLabel), _cmp_desc);

    /* Allocate output arrays (n+2 points: start and end of curve) */
    int max_pts = n + 2;
    ROCCurve *rc = (ROCCurve *)calloc(1, sizeof(ROCCurve));
    if (!rc) { free(sorted); return NULL; }

    rc->fpr        = (double *)malloc((size_t)max_pts * sizeof(double));
    rc->tpr        = (double *)malloc((size_t)max_pts * sizeof(double));
    rc->thresholds = (double *)malloc((size_t)max_pts * sizeof(double));
    if (!rc->fpr || !rc->tpr || !rc->thresholds) {
        roc_free(rc); free(sorted); return NULL;
    }

    /* Start at (0,0) with threshold = infinity */
    int idx = 0;
    rc->fpr[idx] = 0.0; rc->tpr[idx] = 0.0;
    rc->thresholds[idx] = sorted[0].prob + 1.0;
    idx++;

    int tp = 0, fp = 0;

    for (int i = 0; i < n; i++) {
        /* At each new unique threshold, record a point */
        if (i > 0 && sorted[i].prob < sorted[i-1].prob) {
            double fpr_val = (n_neg > 0) ? (double)fp / n_neg : 0.0;
            double tpr_val = (n_pos > 0) ? (double)tp / n_pos : 0.0;
            rc->fpr[idx] = fpr_val;
            rc->tpr[idx] = tpr_val;
            rc->thresholds[idx] = sorted[i].prob;
            idx++;
        }
        if (sorted[i].label == 1) tp++;
        else                      fp++;
    }
    /* End at (1,1) */
    rc->fpr[idx] = 1.0; rc->tpr[idx] = 1.0;
    rc->thresholds[idx] = sorted[n-1].prob - 1.0;
    idx++;

    rc->n_points = idx;

    /* AUC via trapezoidal rule */
    double auc = 0.0;
    for (int i = 1; i < idx; i++) {
        double dx = rc->fpr[i] - rc->fpr[i-1];
        double avg_y = 0.5 * (rc->tpr[i] + rc->tpr[i-1]);
        auc += dx * avg_y;
    }
    rc->auc = auc;

    free(sorted);
    return rc;
}

void roc_free(ROCCurve *r)
{
    if (!r) return;
    free(r->fpr); free(r->tpr); free(r->thresholds);
    free(r);
}

double auc_score(const int *y_true, const double *probs,
                 int n, int positive_class)
{
    ROCCurve *r = roc_curve(y_true, probs, n, positive_class);
    if (!r) return 0.0;
    double a = r->auc;
    roc_free(r);
    return a;
}

double accuracy_score(const int *y_true, const int *y_pred, int n)
{
    int correct = 0;
    for (int i = 0; i < n; i++) if (y_true[i] == y_pred[i]) correct++;
    return (double)correct / (double)n;
}
