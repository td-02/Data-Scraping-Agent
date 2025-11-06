import os
import csv
import requests
from dotenv import load_dotenv
from transformers import pipeline

# === Load Environment Variables ===
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("❌ GOOGLE_MAPS_API_KEY not found in .env file!")

# === Initialize summarizer ===
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token=HF_API_TOKEN)

print("Device set to use cpu")
print("=== 🧠 Hugging Face Business Intelligence Agent ===")

def summarize_text(text):
    """Summarize the text using Hugging Face."""
    try:
        summary = summarizer(text, max_length=40, min_length=10, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return text[:100] + "..."

def search_places_and_save_to_csv(query):
    """Searches for places using the NEW Google Places API v1 and saves to CSV."""
    print(f"🔍 Searching for {query}...")

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,"
            "places.rating,places.userRatingCount,"
            "places.types,places.nationalPhoneNumber,"
            "places.websiteUri,places.shortFormattedAddress"
        ),
    }
    data = {"textQuery": query}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        print("❌ Error fetching data:", response.text)
        return

    results = response.json().get("places", [])
    if not results:
        print("❌ No results found.")
        return

    # === Prepare CSV output ===
    csv_filename = os.path.expanduser("~/Downloads/business_results.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Business Name",
            "Type",
            "Rating",
            "Total Reviews",
            "Address",
            "Phone",
            "Website",
            "Summary of Best Rated Item",
        ])

        for place in results:
            name = place.get("displayName", {}).get("text", "N/A")
            types = ", ".join(place.get("types", []))
            rating = place.get("rating", "N/A")
            reviews = place.get("userRatingCount", "N/A")
            address = place.get("formattedAddress", "N/A")
            phone = place.get("nationalPhoneNumber", "N/A")
            website = place.get("websiteUri", "N/A")

            # Create a short business summary
            description = f"{name} is a {types} located at {address}. It has a rating of {rating} based on {reviews} reviews."
            summary = summarize_text(description)

            writer.writerow([name, types, rating, reviews, address, phone, website, summary])

    print(f"✅ Results saved to {csv_filename}")

if __name__ == "__main__":
    query = input("Enter business type and area (e.g., cafes near Park Street Kolkata): ")
    search_places_and_save_to_csv(query)
