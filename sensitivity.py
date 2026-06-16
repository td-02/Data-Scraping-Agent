import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from business_suitability_single_file import (
    DEFAULT_SCORE_WEIGHTS,
    DOWNLOADS,
    compute_business_score,
    merge_score_weights,
)


def _perturb_weights(base_weights, multiplier):
    return {key: float(value) * multiplier for key, value in base_weights.items()}


def _scenario_name(key=None, multiplier=1.0, joint=False):
    suffix = "+20%" if multiplier > 1 else "-20%"
    if joint:
        return f"joint_{suffix}"
    return f"{key}_{suffix}"


def run_sensitivity_analysis(results, weights=None, run_id=None, export_path=None):
    """
    Evaluate label stability under independent and joint +/-20% perturbations.

    Returns a DataFrame with one row per business per perturbation. The baseline
    score/label from `results` is preserved so downstream analysis can quantify
    which businesses are robustly suitable versus marginal.
    """
    base_weights = merge_score_weights(weights)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = Path(export_path) if export_path else DOWNLOADS / f"sensitivity_results_{run_id}.csv"

    scenarios = []
    for key in DEFAULT_SCORE_WEIGHTS.keys():
        for multiplier in (0.8, 1.2):
            perturbed = dict(base_weights)
            perturbed[key] = perturbed[key] * multiplier
            scenarios.append((_scenario_name(key=key, multiplier=multiplier), perturbed))

    for multiplier in (0.8, 1.2):
        scenarios.append((_scenario_name(multiplier=multiplier, joint=True), _perturb_weights(base_weights, multiplier)))

    rows = []
    for row in results:
        baseline_score = row.get("score")
        baseline_label = row.get("label")
        for scenario_name, scenario_weights in scenarios:
            score, label = compute_business_score(
                row.get("rating"),
                row.get("user_ratings_total"),
                row.get("competitor_count"),
                row.get("avg_comp_rating"),
                weights=scenario_weights,
                use_trained_model=False,
            )
            rows.append(
                {
                    "business_name": row.get("name"),
                    "scenario": scenario_name,
                    "score": score,
                    "label": label,
                    "baseline_score": baseline_score,
                    "baseline_label": baseline_label,
                    "label_changed": label != baseline_label,
                    "score_delta": None if baseline_score is None else round(score - float(baseline_score), 4),
                    "weights": json.dumps(scenario_weights, sort_keys=True),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(export_path, index=False)
    return df, export_path
