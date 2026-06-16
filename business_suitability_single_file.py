#!/usr/bin/env python3
"""
Improved Business Suitability Single-File Agent

Improvements over prior version:
- Deduplicate places by normalized name + approx location
- Competitor counting uses geographic radius (default 500 m)
- Slightly more lenient and explainable scoring weights
- Cleaner LLM prompt for concise verdict + pros/cons
- Saves CSV to Downloads

Requirements:
pip install python-dotenv requests pandas scipy transformers torch tqdm
"""

import argparse
import csv
import json
import math
import os
import pickle
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import Counter

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=HERE / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

DOWNLOADS = Path.home() / "Downloads"
try:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    _write_probe = DOWNLOADS / ".codex_write_probe"
    with open(_write_probe, "w", encoding="utf-8") as f:
        f.write("")
    _write_probe.unlink(missing_ok=True)
except Exception:
    DOWNLOADS = HERE / "outputs"
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

DETERMINISTIC_VERDICT_MODEL_NAME = "deterministic-verdict-v1"
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-haiku-20241022")
SCORING_MODEL_PATH = HERE / "models" / "scoring_model.pkl"


@dataclass(frozen=True)
class ScoringConfig:
    """Documented defaults for the hand-tuned market viability rubric."""

    rating_cutoffs: tuple[float, float, float] = (4.5, 4.0, 3.5)
    rating_points: tuple[int, int, int] = (3, 2, 1)
    review_cutoffs: tuple[int, int, int] = (2000, 700, 150)
    review_points: tuple[int, int, int] = (3, 2, 1)
    low_competitor_threshold: int = 2
    moderate_competitor_threshold: int = 5
    low_competition_points: int = 2
    moderate_competition_points: int = 1
    high_competition_points: int = -1
    weak_competitor_rating: float = 3.9
    strong_competitor_rating: float = 4.3
    weak_competitor_points: int = 1
    strong_competitor_points: int = -1
    highly_suitable_threshold: float = 7.0
    moderately_suitable_threshold: float = 4.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "rating": 1.0,
            "reviews": 1.0,
            "competition": 1.0,
            "competitor_quality": 1.0,
        }
    )


DEFAULT_SCORING_CONFIG = ScoringConfig()

DEFAULT_SCORE_WEIGHTS = {
    "rating": 1.0,
    "reviews": 1.0,
    "competition": 1.0,
    "competitor_quality": 1.0,
}

DEFAULT_LABEL_THRESHOLDS = {
    "highly": 7.0,
    "moderately": 4.0,
}


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def merge_score_weights(weights=None):
    merged = dict(DEFAULT_SCORING_CONFIG.weights)
    if weights:
        for key, value in weights.items():
            if key in merged and value is not None:
                merged[key] = float(value)
    return merged


def build_model_features(rating, reviews, competitor_count, avg_comp_rating) -> list[float]:
    """Return the numeric feature vector used by the optional trained scorer."""
    return [
        0.0 if rating is None else float(rating),
        0.0 if reviews is None else float(reviews),
        0.0 if competitor_count is None else float(competitor_count),
        0.0 if avg_comp_rating is None else float(avg_comp_rating),
    ]


def load_scoring_model(model_path: Path | str = SCORING_MODEL_PATH):
    """Load the optional trained scoring model if it exists."""
    path = Path(model_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------
# HTTP helpers
# ---------------------------
def safe_post(url, headers=None, json_payload=None, timeout=30, retries=2, backoff=1.2):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            if attempt == retries:
                raise
            time.sleep(backoff * (attempt + 1))
    raise last_exc

# ---------------------------
# Places API wrapper (text search v1)
# ---------------------------
PLACES_SEARCH_BASE = "https://places.googleapis.com/v1/places:searchText"
SAFE_FIELD_MASK = (
    "places.displayName,places.formattedAddress,places.rating,"
    "places.userRatingCount,places.location,places.types"
)

def fetch_places_text_search(query: str, page_limit: int = 2):
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY missing in .env")
    url = f"{PLACES_SEARCH_BASE}?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json", "X-Goog-FieldMask": SAFE_FIELD_MASK}
    payload = {"textQuery": query}
    places = []
    for page in range(page_limit):
        resp = safe_post(url, headers=headers, json_payload=payload)
        data = resp.json()
        page_places = data.get("places", [])
        places.extend(page_places)
        next_token = data.get("nextPageToken") or data.get("token")
        if not next_token:
            break
        payload = {"pageToken": next_token}
        time.sleep(1.0)
    return places

# ---------------------------
# Helpers: normalize, dedupe, haversine
# ---------------------------
def normalize_place(place):
    # parse name
    name = None
    if isinstance(place.get("displayName"), dict):
        name = place["displayName"].get("text")
    if not name:
        name = place.get("name") or place.get("displayName") or "N/A"
    types = ", ".join(place.get("types", [])) if place.get("types") else "Unknown"
    rating = place.get("rating")
    reviews = place.get("userRatingCount") or 0
    address = place.get("formattedAddress") or place.get("shortFormattedAddress") or "N/A"
    lat = None; lng = None
    if place.get("location"):
        lat = place["location"].get("latitude"); lng = place["location"].get("longitude")
    return {
        "name": name.strip(),
        "types": types,
        "rating": float(rating) if rating not in (None, "") else None,
        "user_ratings_total": int(reviews) if reviews else 0,
        "address": address,
        "lat": lat,
        "lng": lng,
        "raw": place
    }

def dedupe_places(places, same_name_tol=0.0005):
    """
    Deduplicate by normalized lowercase name and close lat/lng.
    same_name_tol is ~0.0005 degrees (~50m), adjust if needed.
    """
    out = []
    seen = []
    for p in places:
        n = p["name"].lower()
        lat = p.get("lat"); lng = p.get("lng")
        found = False
        for s in seen:
            # if same name and lat/lng close enough, treat as same
            if s["name"] == n:
                s_lat = s.get("lat"); s_lng = s.get("lng")
                if lat is None or s_lat is None:
                    # fallback to name-only dedupe
                    found = True
                    break
                if abs(lat - s_lat) <= same_name_tol and abs(lng - s_lng) <= same_name_tol:
                    found = True
                    break
        if not found:
            seen.append({"name": n, "lat": lat, "lng": lng})
            out.append(p)
    return out

def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in km
    if None in (lat1, lon1, lat2, lon2):
        return float("inf")
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ---------------------------
# Scoring (adjustable)
# ---------------------------
def label_for_score(score, config: ScoringConfig = DEFAULT_SCORING_CONFIG):
    """Map a 0-10 suitability score to the existing label scale."""
    if score >= config.highly_suitable_threshold:
        return "Highly suitable"
    if score >= config.moderately_suitable_threshold:
        return "Moderately suitable"
    return "Not recommended"


def _score_from_cutoffs(value, cutoffs, points):
    for cutoff, point in zip(cutoffs, points):
        if value >= cutoff:
            return point
    return 0


def compute_business_score(
    rating,
    reviews,
    competitor_count,
    avg_comp_rating,
    weights=None,
    config: ScoringConfig = DEFAULT_SCORING_CONFIG,
    use_trained_model: bool = True,
    model_path: Path | str = SCORING_MODEL_PATH,
):
    """
    Formal parameterization of the business suitability score.

    We model market viability as:

        V(r, n, c, q; w) = w_r * phi_r(r) + w_n * phi_n(n) + w_c * phi_c(c) + w_q * phi_q(q)

    where:
    - r is the Google rating
    - n is the review count
    - c is the competitor count inside the local radius
    - q is the average competitor rating inside the local radius
    - w = {rating, reviews, competition, competitor_quality} is a configurable weight vector

    The phi terms are the default piecewise basis functions induced by the current hardcoded
    thresholds:
    - phi_r(r): 3/2/1/0 for rating >= 4.5 / 4.0 / 3.5 / otherwise
    - phi_n(n): 3/2/1/0 for reviews >= 2000 / 700 / 150 / otherwise
    - phi_c(c): 2/1/-1 for competitor_count <= 2 / <= 5 / otherwise
    - phi_q(q): 1/-1/0 for avg competitor rating < 3.9 / >= 4.3 / otherwise

    If `models/scoring_model.pkl` exists and `use_trained_model=True`, the fitted logistic
    model is used instead and its predicted probability is scaled to the 0-10 score range.
    Otherwise the default weight vector preserves the existing scoring behavior.
    """
    if use_trained_model:
        model = load_scoring_model(model_path)
        if model is not None:
            features = [build_model_features(rating, reviews, competitor_count, avg_comp_rating)]
            probability = float(model.predict_proba(features)[0][1])
            score = round(probability * 10.0, 4)
            return score, label_for_score(score, config)

    weights = merge_score_weights(weights)
    rating = 0.0 if rating is None else float(rating)
    reviews = 0 if reviews is None else int(reviews)
    competitor_count = 0 if competitor_count is None else int(competitor_count)
    avg_comp_rating = 0.0 if avg_comp_rating is None else float(avg_comp_rating)

    rating_term = _score_from_cutoffs(rating, config.rating_cutoffs, config.rating_points)
    review_term = _score_from_cutoffs(reviews, config.review_cutoffs, config.review_points)
    if competitor_count <= config.low_competitor_threshold:
        competition_term = config.low_competition_points
    elif competitor_count <= config.moderate_competitor_threshold:
        competition_term = config.moderate_competition_points
    else:
        competition_term = config.high_competition_points
    quality_term = 0
    if competitor_count > 0:
        if avg_comp_rating < config.weak_competitor_rating:
            quality_term = config.weak_competitor_points
        elif avg_comp_rating >= config.strong_competitor_rating:
            quality_term = config.strong_competitor_points

    score = (
        weights["rating"] * rating_term
        + weights["reviews"] * review_term
        + weights["competition"] * competition_term
        + weights["competitor_quality"] * quality_term
    )
    score = max(0.0, round(float(score), 4))
    label = label_for_score(score, config)
    return score, label


def compute_morans_i(results, local_radius_m):
    valid = []
    for row in results:
        if row.get("lat") is None or row.get("lng") is None:
            continue
        if row.get("score") is None:
            continue
        valid.append(row)

    n = len(valid)
    if n < 3:
        return float("nan")

    mean_score = statistics.mean([float(r["score"]) for r in valid])
    denom = sum((float(r["score"]) - mean_score) ** 2 for r in valid)
    if denom == 0:
        return float("nan")

    numerator = 0.0
    s0 = 0.0
    for i, row_i in enumerate(valid):
        for j, row_j in enumerate(valid):
            if i == j:
                continue
            dist_km = haversine_km(row_i["lat"], row_i["lng"], row_j["lat"], row_j["lng"])
            if dist_km == float("inf") or dist_km * 1000 > local_radius_m:
                continue
            weight = 1.0 / max(dist_km, 1e-6)
            s0 += weight
            numerator += weight * (float(row_i["score"]) - mean_score) * (float(row_j["score"]) - mean_score)

    if s0 == 0:
        return float("nan")
    return round((n / s0) * (numerator / denom), 4)


def compute_competitor_hhi(local_comps):
    rated = [round(float(c["rating"]), 1) for c in local_comps if c.get("rating") is not None]
    if not rated:
        return 0.0
    total = len(rated)
    counts = Counter(rated)
    return round(sum((count / total) ** 2 for count in counts.values()), 4)

# ---------------------------
# Verdict generation
# ---------------------------
def generate_verdict(name, rating, score, label, competitor_count, avg_comp_rating, local_radius_m):
    """Generate a deterministic suitability verdict from structured scoring fields."""
    density = "Low" if competitor_count <= 2 else "Moderate" if competitor_count <= 5 else "High"
    rating_signal = "strong" if (rating or 0.0) >= 4.3 else "adequate" if (rating or 0.0) >= 3.8 else "weak"
    competitor_signal = (
        "weak competitor quality"
        if competitor_count and avg_comp_rating < 3.9
        else "strong competitor quality"
        if competitor_count and avg_comp_rating >= 4.3
        else "mixed competitor quality"
        if competitor_count
        else "no local competitor signal"
    )
    action = (
        "Recommend market entry."
        if label == "Highly suitable"
        else "Consider entry with local validation."
        if label == "Moderately suitable"
        else "Do not prioritize market entry without stronger evidence."
    )
    return (
        f"Score: {score}/10 ({label}). "
        f"{density} competitor density ({competitor_count} within {local_radius_m}m), "
        f"{rating_signal} rating signal ({rating or 0.0}), {competitor_signal} "
        f"(avg {avg_comp_rating}). {action}"
    )


def reason_with_anthropic(payload):
    """Generate an optional LLM verdict from structured inputs using Anthropic."""
    if not ANTHROPIC_API_KEY:
        return "LLM verdict unavailable: ANTHROPIC_API_KEY missing."

    body = {
        "model": ANTHROPIC_MODEL_NAME,
        "max_tokens": 220,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Return a concise local market-entry verdict from this JSON. "
                            "Use 3 sentences maximum and do not invent unavailable facts.\n"
                            f"{json.dumps(payload, sort_keys=True, default=str)}"
                        ),
                    }
                ],
            }
        ],
    }
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
    }
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=45)
        response.raise_for_status()
        data = response.json()
        content = data.get("content", [])
        if content and isinstance(content[0], dict):
            return str(content[0].get("text", "")).strip()
        return str(data).strip()
    except Exception as e:
        return f"LLM error: {e}"

# ---------------------------
# Export CSV
# ---------------------------
def write_run_manifest(csv_path, config):
    manifest_path = csv_path.with_name(f"{csv_path.stem}_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    return manifest_path


def export_results(rows, prefix="suitability_results", run_id=None, manifest_config=None):
    ts = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.csv"
    path = DOWNLOADS / filename
    headers = ["Business Name","Type","Rating","Total Reviews","Address","Lat","Lng",
               "Local Competitors","Avg Local Comp Rating","Score","Label","Verdict",
               "Moran I Approx","Competitor HHI"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([
                r["name"], r["types"], r["rating"], r["user_ratings_total"], r["address"],
                r["lat"], r["lng"], r["competitor_count"], r["avg_comp_rating"],
                r["score"], r["label"], r["verdict"], r.get("moran_i_approx"), r.get("competitor_hhi")
            ])
    if manifest_config is not None:
        manifest_payload = dict(manifest_config)
        manifest_payload["output_csv"] = str(path)
        write_run_manifest(path, manifest_payload)
    return path

# ---------------------------
# Main evaluation pipeline
# ---------------------------
def evaluate_query(query, pages=2, local_radius_m=500, weights=None, seed=None, run_analyses=True, use_llm=False):
    set_seed(seed)
    weights = merge_score_weights(weights)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = fetch_places_text_search(query, page_limit=pages)
    if not raw:
        print("No results.")
        return []
    normalized = [normalize_place(p) for p in raw]
    # dedupe by name+location
    normalized = dedupe_places(normalized)
    results = []
    for i, place in enumerate(tqdm(normalized, desc="Evaluating")):
        # calculate local competitors within radius
        local_comps = []
        for j, other in enumerate(normalized):
            if i == j: continue
            dist_km = haversine_km(place.get("lat"), place.get("lng"), other.get("lat"), other.get("lng"))
            if dist_km == float("inf"): 
                # if location missing, treat as non-local
                continue
            if dist_km * 1000 <= local_radius_m:
                local_comps.append(other)
        comp_count = len(local_comps)
        avg_comp_rating = round(statistics.mean([c["rating"] or 0.0 for c in local_comps]) if local_comps else 0.0, 2)
        score, label = compute_business_score(
            place["rating"],
            place["user_ratings_total"],
            comp_count,
            avg_comp_rating,
            weights=weights,
        )
        verdict_payload = {
            "business_name": place["name"],
            "query": query,
            "rating": place["rating"],
            "review_count": place["user_ratings_total"],
            "score": score,
            "label": label,
            "competitor_count": comp_count,
            "avg_competitor_rating": avg_comp_rating,
            "local_radius_m": local_radius_m,
        }
        verdict = (
            reason_with_anthropic(verdict_payload)
            if use_llm
            else generate_verdict(
                place["name"],
                place["rating"],
                score,
                label,
                comp_count,
                avg_comp_rating,
                local_radius_m,
            )
        )
        competitor_hhi = compute_competitor_hhi(local_comps)
        row = {
            "name": place["name"], "types": place["types"], "rating": place["rating"],
            "user_ratings_total": place["user_ratings_total"], "address": place["address"],
            "lat": place["lat"], "lng": place["lng"], "competitor_count": comp_count,
            "avg_comp_rating": avg_comp_rating, "score": score, "label": label, "verdict": verdict,
            "competitor_hhi": competitor_hhi
        }
        results.append(row)
        time.sleep(0.2)
    morans_i = compute_morans_i(results, local_radius_m)
    for row in results:
        row["moran_i_approx"] = morans_i

    print(f"\nSpatial Moran's I approximation: {morans_i}")
    print("Per-business competitor HHI computed and added to the CSV output.")
    try:
        from spatial import analyze_spatial_clusters

        spatial_analysis = analyze_spatial_clusters(results, local_radius_m=local_radius_m, moran_i=morans_i)
    except Exception as e:
        spatial_analysis = {"moran_i": morans_i, "error": str(e)}

    print("\nSpatial cluster analysis:")
    print(json.dumps(spatial_analysis, indent=2, default=str))

    manifest_config = {
        "query": query,
        "pages": pages,
        "radius_m": local_radius_m,
        "weights": weights,
        "seed": seed,
        "timestamp": run_id,
        "model_name": ANTHROPIC_MODEL_NAME if use_llm else DETERMINISTIC_VERDICT_MODEL_NAME,
        "llm_enabled": use_llm,
        "output_csv": None,
        "spatial_analysis": spatial_analysis,
        "competitor_hhi_definition": "sum of squared shares of rounded competitor ratings to 1 decimal place",
    }

    out = export_results(results, run_id=run_id, manifest_config=manifest_config)
    print(f"\nSaved CSV to: {out}")

    if run_analyses:
        try:
            from sensitivity import run_sensitivity_analysis

            sensitivity_df, sensitivity_out = run_sensitivity_analysis(
                results,
                weights=weights,
                run_id=run_id,
            )
            sensitivity_manifest = {
                "query": query,
                "pages": pages,
                "radius_m": local_radius_m,
                "weights": weights,
                "seed": seed,
                "timestamp": run_id,
                "model_name": ANTHROPIC_MODEL_NAME if use_llm else DETERMINISTIC_VERDICT_MODEL_NAME,
                "output_csv": str(sensitivity_out),
                "source_csv": str(out),
                "analysis_type": "sensitivity",
                "rows": int(len(sensitivity_df)),
            }
            write_run_manifest(sensitivity_out, sensitivity_manifest)
            write_run_manifest(out, {**manifest_config, "output_csv": str(out), "sensitivity_csv": str(sensitivity_out)})
        except Exception as e:
            print(f"Warning: sensitivity analysis failed: {e}")
        try:
            from baseline import compare_baselines

            compare_baselines(results)
        except Exception as e:
            print(f"Warning: baseline comparison failed: {e}")
    return results

# ---------------------------
# CLI
# ---------------------------
def main():
    print("=== Business Suitability Hybrid Agent (Improved) ===")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--llm", action="store_true", help="Use Anthropic for optional LLM verdicts")
    args, _ = parser.parse_known_args()

    set_seed(args.seed)

    q = input("Enter business query (e.g. 'cafes near Park Street Kolkata'): ").strip()
    if not q:
        print("Enter a non-empty query.")
        return
    pages_in = input("Pages to fetch (1-3, default 2): ").strip()
    try: pages = max(1, min(3, int(pages_in))) 
    except: pages = 2
    rad_in = input("Local radius in meters for 'local competitors' (default 500): ").strip()
    try: rad = max(100, min(2000, int(rad_in))) 
    except: rad = 500
    results = evaluate_query(q, pages=pages, local_radius_m=rad, seed=args.seed, use_llm=args.llm)
    if results:
        print("\nTop 5 results:")
        for r in results[:5]:
            print(f"- {r['name']} | Score: {r['score']} | {r['label']}")
            print(f"  Local comps: {r['competitor_count']} | Avg comp rating: {r['avg_comp_rating']}")
            print(f"  Verdict short: {r['verdict'].splitlines()[0] if r['verdict'] else 'N/A'}\n")

if __name__ == "__main__":
    main()
