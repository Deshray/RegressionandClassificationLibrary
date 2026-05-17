/*
 * linalg.c
 * --------
 * Implementation of the matrix operations declared in linalg.h.
 * Everything operates on flat row-major double arrays.
 *
 * The core routine is mat_inv_logdet, which performs Gauss-Jordan
 * elimination with partial pivoting and accumulates the log-determinant
 * as a side-product.  mat_inv calls this with a NULL logdet pointer.
 */

#include "linalg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------ */
/* mat_mul  C = A @ B                                                  */
/* ------------------------------------------------------------------ */
void mat_mul(const double *A, const double *B, double *C,
             int m, int k, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int l = 0; l < k; l++)
                s += A[i*k + l] * B[l*n + j];
            C[i*n + j] = s;
        }
    }
}

/* ------------------------------------------------------------------ */
/* mat_transpose  B = A^T                                              */
/* ------------------------------------------------------------------ */
void mat_transpose(const double *A, double *B, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            B[j*m + i] = A[i*n + j];
}

/* ------------------------------------------------------------------ */
/* mat_inv_logdet  A -> A^{-1},  *logdet = log|det(A)|               */
/* ------------------------------------------------------------------ */
int mat_inv_logdet(double *A, int n, double *logdet)
{
    /*
     * Augment [A | I] to a (n x 2n) working matrix, apply
     * Gauss-Jordan with partial pivoting, then extract A^{-1}
     * from the right half.  Accumulate sign and log-magnitude
     * of pivots to produce  log|det(A)|.
     */
    int w = 2 * n;
    double *aug = (double *)calloc((size_t)n * (size_t)w, sizeof(double));
    if (!aug) return -1;

    /* Fill augmented matrix */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug[i*w + j] = A[i*n + j];
        aug[i*w + n + i] = 1.0;   /* identity in right half */
    }

    double log_det = 0.0;
    int sign = 1;

    for (int col = 0; col < n; col++) {
        /* Partial pivoting: find row with largest |element| in this col */
        int pivot = col;
        for (int row = col + 1; row < n; row++) {
            if (fabs(aug[row*w + col]) > fabs(aug[pivot*w + col]))
                pivot = row;
        }

        if (fabs(aug[pivot*w + col]) < 1e-14) {
            free(aug);
            return -1;   /* singular matrix */
        }

        /* Swap rows col and pivot */
        if (pivot != col) {
            for (int j = 0; j < w; j++) {
                double tmp          = aug[col*w   + j];
                aug[col*w   + j]    = aug[pivot*w + j];
                aug[pivot*w + j]    = tmp;
            }
            sign = -sign;  /* row swap flips sign of determinant */
        }

        /* Accumulate log-determinant from the diagonal pivot */
        log_det += log(fabs(aug[col*w + col]));

        /* Normalise pivot row */
        double diag = aug[col*w + col];
        for (int j = 0; j < w; j++)
            aug[col*w + j] /= diag;

        /* Eliminate all other rows in this column */
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = aug[row*w + col];
            for (int j = 0; j < w; j++)
                aug[row*w + j] -= factor * aug[col*w + j];
        }
    }

    /* Extract A^{-1} from right half of augmented matrix */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i*n + j] = aug[i*w + n + j];

    if (logdet) *logdet = log_det;   /* caller may pass NULL */

    free(aug);
    return 0;
}

/* ------------------------------------------------------------------ */
/* mat_inv  A -> A^{-1}  (discards log-determinant)                   */
/* ------------------------------------------------------------------ */
int mat_inv(double *A, int n)
{
    return mat_inv_logdet(A, n, NULL);
}

/* ------------------------------------------------------------------ */
/* mat_vec_mul  y = A @ x                                              */
/* ------------------------------------------------------------------ */
void mat_vec_mul(const double *A, const double *x, double *y,
                 int m, int n)
{
    for (int i = 0; i < m; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++)
            s += A[i*n + j] * x[j];
        y[i] = s;
    }
}

/* ------------------------------------------------------------------ */
/* mat_copy  dst = src  (square n x n)                                 */
/* ------------------------------------------------------------------ */
void mat_copy(const double *src, double *dst, int n)
{
    memcpy(dst, src, (size_t)n * (size_t)n * sizeof(double));
}

/* ------------------------------------------------------------------ */
/* vec_copy  dst = src  (flat array of length len)                     */
/* ------------------------------------------------------------------ */
void vec_copy(const double *src, double *dst, int len)
{
    memcpy(dst, src, (size_t)len * sizeof(double));
}

/* ------------------------------------------------------------------ */
/* vec_dot  a . b                                                       */
/* ------------------------------------------------------------------ */
double vec_dot(const double *a, const double *b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}
