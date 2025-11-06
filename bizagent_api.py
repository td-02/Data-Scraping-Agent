import os
import csv
import googlemaps
from transformers import pipeline
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ==============================
# Load environment variables
# ==============================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

# ==============================
# Initialize clients
# ==============================
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ==============================
# Output CSV path (Downloads folder)
# ==============================
downloads_path = str(Path.home() / "Downloads")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(downloads_path, f"business_data_{timestamp}.csv")

# ==============================
# Helper function to summarize
# ==============================
def generate_best_item_summary(place_name, place_type):
    try:
        prompt = (
            f"Write a short, catchy one-sentence description of the most popular or best-rated item "
            f"served or offered at {place_name}, a {place_type} in Kolkata."
        )
        result = summarizer(prompt, max_length=40, min_length=15, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Summary unavailable ({str(e)})"

# ==============================
# Google Maps search + CSV writing
# ==============================
def search_places_and_save_to_csv(query):
    print(f"🔍 Searching for {query}...")
    places_result = gmaps.places(query=query)

    if not places_result.get("results"):
        print("❌ No results found.")
        return

    # Prepare CSV file with headers
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Business Name",
            "Business Type",
            "Rating",
            "Total Reviews",
            "Address",
            "Best Rated Item"
        ])

        # Iterate through all results
        for place in places_result["results"]:
            name = place.get("name", "N/A")
            types = ", ".join(place.get("types", []))
            rating = place.get("rating", "N/A")
            reviews = place.get("user_ratings_total", "N/A")
            address = place.get("vicinity", "N/A")

            best_item_summary = generate_best_item_summary(name, types)
            writer.writerow([name, types, rating, reviews, address, best_item_summary])

    print(f"✅ Data saved successfully to {csv_filename}")


# ==============================
# Main Program
# ==============================
if __name__ == "__main__":
    print("=== 🧠 Hugging Face Business Intelligence Agent ===")
    query = input("Enter business type and area (e.g., cafes near Park Street Kolkata): ")

    if not query.strip():
        print("❌ Please enter a valid query.")
    else:
        search_places_and_save_to_csv(query)
