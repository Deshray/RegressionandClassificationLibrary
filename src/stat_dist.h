/*
 * stat_dist.h
 * -----------
 * Distribution functions needed to compute p-values and confidence-interval
 * quantiles exactly as described in ISLP Chapters 3 and 4.
 *
 * Functions
 * ---------
 *   normal_cdf(x)              Phi(x), standard normal CDF                [Ch.4 z-stats]
 *   t_pvalue_two_sided(t, df)  2*P(T_{df} > |t|)                         [Ch.3 Eq 3.14]
 *   f_pvalue(F, df1, df2)      P(F_{df1,df2} > F)                        [Ch.3 Eq 3.23]
 *   t_quantile(p, df)          x s.t. P(T_{df} <= x) = p  (bisection)    [Ch.3 CI/PI]
 */

#ifndef STAT_DIST_H
#define STAT_DIST_H

/*
 * Standard normal CDF  Phi(x) = P(Z <= x),  Z ~ N(0,1)
 * Uses the complementary error function from <math.h>.
 */
double normal_cdf(double x);

/*
 * Two-sided p-value for a t-statistic with `df` degrees of freedom.
 *   returns  2 * P(T_{df} > |t|)
 * Used for every coefficient test in linreg (ISLP 3.1.2) and logreg (ISLP 4.3.2).
 *
 * Implementation: I(df/(df+t²); df/2, 1/2)  via regularised incomplete beta.
 */
double t_pvalue_two_sided(double t, double df);

/*
 * Upper-tail p-value for an F-statistic with df1 numerator and df2 denominator df.
 *   returns  P(F_{df1,df2} > F_stat)
 * Used for the overall model F-test in multiple linear regression (ISLP 3.2.2 Eq 3.23).
 *
 * Implementation: I(df2/(df2 + df1*F); df2/2, df1/2)  via regularised incomplete beta.
 */
double f_pvalue(double F_stat, int df1, int df2);

/*
 * Inverse t-distribution (quantile).
 *   returns x s.t. P(T_{df} <= x) = p
 * Used for 95% confidence intervals  beta_hat +/- t_{n-p-1, 0.025} * SE  (ISLP 3.9–3.11)
 * and for the prediction-interval half-width (ISLP 3.2.2).
 * Implemented via bisection on t_pvalue_two_sided.
 */
double t_quantile(double p, double df);

#endif /* STAT_DIST_H */
