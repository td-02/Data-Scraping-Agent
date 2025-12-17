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
pip install python-dotenv requests pandas transformers torch tqdm
"""

import os
import time
import csv
import math
import statistics
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# transformers import
try:
    from transformers import pipeline
except Exception as e:
    raise RuntimeError("Install transformers + torch: pip install transformers torch") from e

# ---------------------------
# Config
# ---------------------------
HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=HERE / ".env")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing in .env")

DOWNLOADS = Path.home() / "Downloads"
DOWNLOADS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/flan-t5-base"
print("Initializing LLM (may download first run)...")
hf_kwargs = {}
if HF_API_TOKEN:
    hf_kwargs["token"] = HF_API_TOKEN
try:
    reasoner = pipeline("text2text-generation", model=MODEL_NAME, **hf_kwargs)
except Exception as e:
    print("Warning: model init with token failed; trying without token...", e)
    reasoner = pipeline("text2text-generation", model=MODEL_NAME)
print("LLM ready (CPU)")

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
def compute_business_score(rating, reviews, competitor_count, avg_comp_rating):
    """
    Slightly more lenient weights:
    - rating: up to 3 pts
    - reviews: up to 3 pts
    - low competition gives +2, moderate +1, high -1
    - weak competitors (avg < 3.9) gives +1
    """
    score = 0
    # rating
    if rating is None:
        rating = 0.0
    if rating >= 4.5:
        score += 3
    elif rating >= 4.0:
        score += 2
    elif rating >= 3.5:
        score += 1
    # reviews
    if reviews >= 2000:
        score += 3
    elif reviews >= 700:
        score += 2
    elif reviews >= 150:
        score += 1
    # competition density
    if competitor_count <= 2:
        score += 2
    elif competitor_count <= 5:
        score += 1
    else:
        score -= 1
    # competitor strength
    if competitor_count > 0:
        if avg_comp_rating < 3.9:
            score += 1
        elif avg_comp_rating >= 4.3:
            score -= 1
    if score < 0:
        score = 0
    # map to label
    if score >= 7:
        label = "Highly suitable"
    elif score >= 4:
        label = "Moderately suitable"
    else:
        label = "Not recommended"
    return score, label

# ---------------------------
# LLM reasoner prompt (improved)
# ---------------------------
def build_reasoner_prompt(name, query_location, score_label, score, local_comps):
    comp_lines = ""
    for c in local_comps[:8]:
        comp_lines += f"- {c['name']} (rating: {c['rating']}, reviews: {c['user_ratings_total']})\n"
    if not comp_lines:
        comp_lines = "None listed.\n"
    prompt = f"""You are a concise local business consultant.

Business name: {name}
Location (user query): {query_location}
Preliminary score: {score_label} (score {score}/10)

Nearby competitors (within chosen radius):
{comp_lines}

Task: Based on the info above, in 5-8 short lines:
1) One-line verdict: Recommended / Not recommended / Consider with caveats.
2) 3 short pros.
3) 3 short cons/risks.
4) 3 practical next steps (what the owner should check or do next).

Keep output brief, numbered, action-oriented.
"""
    return prompt

def reason_with_llm(name, query_location, score_label, score, local_comps):
    prompt = build_reasoner_prompt(name, query_location, score_label, score, local_comps)
    try:
        out = reasoner(prompt, max_length=280, do_sample=False)
        first = out[0]
        text = first.get("generated_text") or first.get("text") or str(first)
        return text.strip()
    except Exception as e:
        return f"LLM error: {e}"

# ---------------------------
# Export CSV
# ---------------------------
def export_results(rows, prefix="suitability_results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.csv"
    path = DOWNLOADS / filename
    headers = ["Business Name","Type","Rating","Total Reviews","Address","Lat","Lng",
               "Local Competitors","Avg Local Comp Rating","Score","Label","Verdict"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([
                r["name"], r["types"], r["rating"], r["user_ratings_total"], r["address"],
                r["lat"], r["lng"], r["competitor_count"], r["avg_comp_rating"],
                r["score"], r["label"], r["verdict"]
            ])
    return path

# ---------------------------
# Main evaluation pipeline
# ---------------------------
def evaluate_query(query, pages=2, local_radius_m=500):
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
        score, label = compute_business_score(place["rating"], place["user_ratings_total"], comp_count, avg_comp_rating)
        verdict = reason_with_llm(place["name"], query, label, score, local_comps)
        row = {
            "name": place["name"], "types": place["types"], "rating": place["rating"],
            "user_ratings_total": place["user_ratings_total"], "address": place["address"],
            "lat": place["lat"], "lng": place["lng"], "competitor_count": comp_count,
            "avg_comp_rating": avg_comp_rating, "score": score, "label": label, "verdict": verdict
        }
        results.append(row)
        time.sleep(0.2)
    out = export_results(results)
    print(f"\nSaved CSV to: {out}")
    return results

# ---------------------------
# CLI
# ---------------------------
def main():
    print("=== Business Suitability Hybrid Agent (Improved) ===")
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
    results = evaluate_query(q, pages=pages, local_radius_m=rad)
    if results:
        print("\nTop 5 results:")
        for r in results[:5]:
            print(f"- {r['name']} | Score: {r['score']} | {r['label']}")
            print(f"  Local comps: {r['competitor_count']} | Avg comp rating: {r['avg_comp_rating']}")
            print(f"  Verdict short: {r['verdict'].splitlines()[0] if r['verdict'] else 'N/A'}\n")

if __name__ == "__main__":
    main()
