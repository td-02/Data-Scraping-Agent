"""Spatial analysis helpers for suitability result sets."""

from __future__ import annotations

import math
import statistics
from typing import Any

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return haversine distance in kilometers."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _global_morans_i(valid: list[dict[str, Any]], local_radius_m: int) -> float:
    """Compute a distance-weighted global Moran's I approximation."""
    n = len(valid)
    if n < 3:
        return float("nan")

    mean_score = statistics.mean(float(row["score"]) for row in valid)
    denom = sum((float(row["score"]) - mean_score) ** 2 for row in valid)
    if denom == 0:
        return float("nan")

    numerator = 0.0
    weight_sum = 0.0
    for i, row_i in enumerate(valid):
        for j, row_j in enumerate(valid):
            if i == j:
                continue
            distance_km = _haversine_km(row_i["lat"], row_i["lng"], row_j["lat"], row_j["lng"])
            if distance_km * 1000 > local_radius_m:
                continue
            weight = 1.0 / max(distance_km, 1e-6)
            weight_sum += weight
            numerator += weight * (float(row_i["score"]) - mean_score) * (float(row_j["score"]) - mean_score)

    if weight_sum == 0:
        return float("nan")
    return round((n / weight_sum) * (numerator / denom), 4)


def _local_moran_scores(valid: list[dict[str, Any]], local_radius_m: int) -> list[tuple[float, float]]:
    """Return `(local_moran_proxy, score)` pairs for bucket-level analysis."""
    if len(valid) < 3:
        return []

    mean_score = statistics.mean(float(row["score"]) for row in valid)
    variance = sum((float(row["score"]) - mean_score) ** 2 for row in valid) / len(valid)
    if variance == 0:
        return []

    pairs: list[tuple[float, float]] = []
    for i, row_i in enumerate(valid):
        weighted_neighbor_deviation = 0.0
        for j, row_j in enumerate(valid):
            if i == j:
                continue
            distance_km = _haversine_km(row_i["lat"], row_i["lng"], row_j["lat"], row_j["lng"])
            if distance_km * 1000 > local_radius_m:
                continue
            weight = 1.0 / max(distance_km, 1e-6)
            weighted_neighbor_deviation += weight * (float(row_j["score"]) - mean_score)
        local_i = ((float(row_i["score"]) - mean_score) / variance) * weighted_neighbor_deviation
        pairs.append((round(local_i, 4), float(row_i["score"])))
    return pairs


def analyze_spatial_clusters(
    results: list[dict[str, Any]],
    local_radius_m: int = 500,
    moran_threshold: float = 0.3,
    moran_i: float | None = None,
) -> dict[str, Any]:
    """Analyze high/low local spatial clustering and compare score distributions."""
    valid = [
        row
        for row in results
        if row.get("lat") is not None and row.get("lng") is not None and row.get("score") is not None
    ]
    global_i = _global_morans_i(valid, local_radius_m) if moran_i is None else moran_i
    local_pairs = _local_moran_scores(valid, local_radius_m)

    high_scores = [score for local_i, score in local_pairs if local_i >= moran_threshold]
    low_scores = [score for local_i, score in local_pairs if local_i < moran_threshold]
    bucket_means = {
        "high": round(statistics.mean(high_scores), 4) if high_scores else None,
        "low": round(statistics.mean(low_scores), 4) if low_scores else None,
    }

    p_value = float("nan")
    if mannwhitneyu is not None and high_scores and low_scores:
        try:
            p_value = float(mannwhitneyu(high_scores, low_scores, alternative="two-sided").pvalue)
        except Exception:
            p_value = float("nan")

    return {
        "moran_i": global_i,
        "threshold": moran_threshold,
        "bucket_means": bucket_means,
        "bucket_counts": {"high": len(high_scores), "low": len(low_scores)},
        "p_value": p_value,
        "significant": bool(not math.isnan(p_value) and p_value < 0.05),
    }
