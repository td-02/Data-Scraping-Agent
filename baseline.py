from __future__ import annotations

from collections import namedtuple
import math

import pandas as pd

KendallResult = namedtuple("KendallResult", ["correlation", "pvalue"])

try:
    from scipy.stats import kendalltau as _scipy_kendalltau
except Exception:
    _scipy_kendalltau = None


def _kendall_tau_b(x, y):
    concordant = discordant = ties_x = ties_y = ties_both = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_both += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    tau = float("nan") if denom == 0 else (concordant - discordant) / denom
    return KendallResult(correlation=tau, pvalue=float("nan"))


def _kendalltau(x, y):
    if _scipy_kendalltau is not None:
        try:
            return _scipy_kendalltau(x, y, nan_policy="omit")
        except Exception:
            pass
    return _kendall_tau_b(list(x), list(y))


def compare_baselines(results):
    """
    Compare two naive baselines against the rubric-based ranking.

    Baseline A: rank by raw Google rating only.
    Baseline B: rank by review count only.
    """
    df = pd.DataFrame(results).copy()
    if df.empty:
        out = pd.DataFrame(columns=["baseline", "kendall_tau", "p_value"])
        print("\nBaseline comparison: no results available.")
        return out

    df["rubric_rank"] = df["score"].rank(method="average", ascending=False)
    df["rating_rank"] = df["rating"].rank(method="average", ascending=False)
    df["reviews_rank"] = df["user_ratings_total"].rank(method="average", ascending=False)

    rating_tau = _kendalltau(df["rubric_rank"], df["rating_rank"])
    reviews_tau = _kendalltau(df["rubric_rank"], df["reviews_rank"])

    out = pd.DataFrame(
        [
            {"baseline": "rating_only", "kendall_tau": rating_tau.correlation, "p_value": rating_tau.pvalue},
            {"baseline": "reviews_only", "kendall_tau": reviews_tau.correlation, "p_value": reviews_tau.pvalue},
        ]
    )

    print("\nBaseline comparison vs rubric ranking:")
    print(out.to_string(index=False))
    return out
