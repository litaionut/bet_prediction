"""
ML Over/Under 2.5: Poisson probabilities.
P(k; lambda) = (lambda^k * e^(-lambda)) / k!
Prob Over 2.5 = 1 - (P(0) + P(1) + P(2)).
"""
import math


def poisson_pmf(k, lam):
    """Probability of exactly k goals when expected goals = lam."""
    if lam < 0 or k < 0:
        return 0.0
    if k > 100:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def prob_over_2_5(lam):
    """Probability of more than 2.5 goals (3+). Equals 1 - (P(0) + P(1) + P(2))."""
    p0 = poisson_pmf(0, lam)
    p1 = poisson_pmf(1, lam)
    p2 = poisson_pmf(2, lam)
    return 1.0 - (p0 + p1 + p2)


def poisson_probabilities(lam):
    """Return dict with P(0), P(1), P(2) and P(Over 2.5)."""
    return {
        "lambda": lam,
        "p0": poisson_pmf(0, lam),
        "p1": poisson_pmf(1, lam),
        "p2": poisson_pmf(2, lam),
        "prob_over_2_5": prob_over_2_5(lam),
    }
