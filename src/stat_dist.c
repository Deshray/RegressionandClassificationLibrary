/*
 * stat_dist.c
 * -----------
 * Implements the distribution functions declared in stat_dist.h.
 *
 * Core algorithm: regularised incomplete beta function I(x; a, b)
 * via Lentz's continued-fraction method (Numerical Recipes §6.4).
 * Both the t-CDF and F-CDF reduce to calls of this single function:
 *
 *   t two-sided p-value:  I(df/(df + t^2);  df/2, 1/2)
 *   F upper-tail p-value: I(df2/(df2 + df1*F);  df2/2, df1/2)
 *
 * The normal CDF uses erfc() from <math.h>, which is faster and more
 * accurate than any polynomial approximation.
 */

#include "stat_dist.h"
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

/* log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b) */
static double log_beta(double a, double b)
{
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/*
 * Continued-fraction part of the regularised incomplete beta.
 * Evaluates  (x^a * (1-x)^b) / (a * B(a,b)) * CF  where CF is
 * Lentz's continued fraction.  Only converges well when
 *   x < (a+1)/(a+b+2);
 * otherwise the caller should use the symmetry identity.
 */
static double beta_cf(double a, double b, double x)
{
    const int    MAXIT = 200;
    const double EPS   = 3.0e-14;
    const double FPMIN = DBL_MIN / EPS;

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c   = 1.0;
    double d   = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= MAXIT; m++) {
        /* Even step */
        double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;

        /* Odd step */
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double delta = d * c;
        h *= delta;

        if (fabs(delta - 1.0) < EPS) break;
    }
    return h;
}

/*
 * Regularised incomplete beta  I(x; a, b) = B(x;a,b)/B(a,b).
 * Returns a value in [0, 1].
 */
static double beta_inc(double a, double b, double x)
{
    if (x < 0.0 || x > 1.0) return -1.0; /* invalid */
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    /* Log of the prefactor  x^a * (1-x)^b / (a * B(a,b)) */
    double lbeta_x = a * log(x) + b * log(1.0 - x)
                     - log(a) - log_beta(a, b);

    /* Choose side of CF for numerical stability */
    if (x < (a + 1.0) / (a + b + 2.0))
        return exp(lbeta_x) * beta_cf(a, b, x);
    else
        return 1.0 - exp(b * log(1.0 - x) + a * log(x)
                         - log(b) - log_beta(a, b))
                         * beta_cf(b, a, 1.0 - x);
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

/*
 * Standard normal CDF  Phi(x).
 * erfc(z) = 1 - erf(z) = 2*P(Z > sqrt(2)*z) for Z~N(0,1).
 * So Phi(x) = erfc(-x/sqrt(2)) / 2.
 */
double normal_cdf(double x)
{
    return 0.5 * erfc(-x / M_SQRT2);
}

/*
 * Two-sided p-value for t-statistic t with df degrees of freedom.
 *   p = 2 * P(T_{df} > |t|) = I(df/(df+t^2); df/2, 1/2)
 *
 * Reference: ISLP Eq (3.14) and surrounding discussion.
 * The t-distribution CDF satisfies:
 *   P(|T| > |t|) = I_x(df/2, 1/2)  where  x = df/(df + t^2).
 */
double t_pvalue_two_sided(double t, double df)
{
    if (df <= 0.0) return 1.0;
    double t2 = t * t;
    double x  = df / (df + t2);
    return beta_inc(df / 2.0, 0.5, x);
}

/*
 * Upper-tail p-value for F-statistic.
 *   P(F_{df1,df2} > F_stat) = I_x(df2/2, df1/2)
 *   where x = df2 / (df2 + df1 * F_stat)
 *
 * Reference: ISLP Eq (3.23) — the F-statistic for the overall model follows
 *   an F-distribution with p and n-p-1 degrees of freedom under H0.
 */
double f_pvalue(double F_stat, int df1, int df2)
{
    if (F_stat <= 0.0 || df1 <= 0 || df2 <= 0) return 1.0;
    double x = (double)df2 / ((double)df2 + (double)df1 * F_stat);
    return beta_inc((double)df2 / 2.0, (double)df1 / 2.0, x);
}

/*
 * Inverse t-distribution quantile via bisection.
 *   Returns x such that P(T_{df} <= x) = p.
 *   Used to compute t_{n-p-1, alpha/2} for confidence intervals (ISLP 3.9-3.11).
 *
 * For a 95% CI, call  t_quantile(0.975, df)  to get the upper critical value.
 * The CI is then  beta_hat +/- t_quantile(0.975, df) * SE(beta_hat).
 */
double t_quantile(double p, double df)
{
    if (p <= 0.0) return -1e15;
    if (p >= 1.0) return  1e15;
    if (p == 0.5) return  0.0;

    /* bisection on the two-sided p-value */
    double lo = 0.0, hi = 1e6;

    /* upper tail: if p > 0.5, we want a positive quantile */
    if (p < 0.5) {
        /* negative quantile by symmetry */
        return -t_quantile(1.0 - p, df);
    }

    /* p > 0.5: find x > 0 such that normal_cdf equiv -> use bisection */
    /* We want: P(T_{df} <= x) = p
     *   = 1 - 0.5 * t_pvalue_two_sided(x, df)  for x > 0
     * => t_pvalue_two_sided(x, df) = 2*(1-p)
     */
    double target = 2.0 * (1.0 - p);

    for (int iter = 0; iter < 200; iter++) {
        double mid = 0.5 * (lo + hi);
        double val = t_pvalue_two_sided(mid, df);
        if (val > target)
            lo = mid;
        else
            hi = mid;
        if (hi - lo < 1e-10) break;
    }
    return 0.5 * (lo + hi);
}
