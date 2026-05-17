/*
 * linalg.h
 * --------
 * Matrix operations used throughout the statistical routines.
 * All matrices are stored as flat row-major double arrays.
 *
 * Naming convention:  A (m x k),  B (k x n),  C (m x n).
 * Element (i,j) of an (m x n) matrix A is  A[i*n + j].
 */

#ifndef LINALG_H
#define LINALG_H

/*
 * C = A @ B   where A is (m x k), B is (k x n), C is (m x n).
 * Caller allocates C of size m*n.
 */
void mat_mul(const double *A, const double *B, double *C,
             int m, int k, int n);

/*
 * B = A^T   where A is (m x n), B is (n x m).
 * Caller allocates B of size n*m.
 */
void mat_transpose(const double *A, double *B, int m, int n);

/*
 * Invert A (n x n) in-place using Gauss-Jordan elimination
 * with partial pivoting.
 * Returns  0  on success,  -1  if A is singular (pivot < 1e-12).
 * A is overwritten with A^{-1}.
 */
int mat_inv(double *A, int n);

/*
 * Invert A (n x n) and return log|det(A)| as a by-product.
 * `logdet` must be a pointer to a double; it is set to log|det(A)|.
 * Returns  0  on success,  -1  if singular.
 * Used by QDA to compute  log|Sigma_k|  while forming Sigma_k^{-1}.
 */
int mat_inv_logdet(double *A, int n, double *logdet);

/*
 * y = A @ x   where A is (m x n), x is (n), y is (m).
 * Caller allocates y of size m.
 */
void mat_vec_mul(const double *A, const double *x, double *y,
                 int m, int n);

/*
 * dst = src   (copy n*n matrix).
 */
void mat_copy(const double *src, double *dst, int n);

/*
 * dst = src   (copy a flat array of length len).
 */
void vec_copy(const double *src, double *dst, int len);

/*
 * dot product of two vectors of length n.
 */
double vec_dot(const double *a, const double *b, int n);

#endif /* LINALG_H */
