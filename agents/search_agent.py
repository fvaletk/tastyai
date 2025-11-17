# agents/search_agent.py

import os
import json
import ast
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_core.tools import tool
from typing import Dict

load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

@tool
def search_recipes(preferences: Dict) -> Dict:
    """
    Search for matching recipes in Pinecone using enriched user preferences.
    Fields used: cuisine, diet, dish, ingredients, meal_type, cooking_time
    """

    query_terms = []

    # Collect fields from preferences
    cuisine = preferences.get("cuisine", "")
    diet = preferences.get("diet", "")
    dish = preferences.get("dish", "")
    meal_type = preferences.get("meal_type", "")
    cooking_time = preferences.get("cooking_time", "")
    ingredients = preferences.get("ingredients", [])

    # Add valid non-unknown fields
    for field in [cuisine, diet, dish, meal_type, cooking_time]:
        if field and field != "unknown":
            query_terms.append(field)

    # Add ingredients list to query
    if ingredients and isinstance(ingredients, list):
        query_terms.extend(ingredients)

    # Fallback if nothing useful was added
    if not query_terms:
        query_terms = ["healthy dinner"]

    query_text = " ".join(query_terms)
    # print(f"🔍 Embedding query: {query_text}")

    try:
        # Generate embedding
        embed_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_vector = embed_resp.data[0].embedding

        # Query Pinecone
        pinecone_response = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )

        # print("🔎 Pinecone matches:", pinecone_response)

        # Extract and clean up metadata
        matches = []
        for match in pinecone_response.matches:
            meta = match.metadata

            def safe_parse_list(value):
                try:
                    return json.loads(value)
                except Exception:
                    try:
                        return ast.literal_eval(value)
                    except Exception:
                        return value

            ingredients = safe_parse_list(meta.get("ingredients", ""))
            directions = safe_parse_list(meta.get("directions", ""))

            matches.append({
                "title": meta.get("title"),
                "link": meta.get("link"),
                "ingredients": ingredients,
                "directions": directions,
                "source": meta.get("source"),
                "score": match.score
            })

        # print("------------------------------------------------------")
        # print("SEARCH RESULT: matches", matches)
        # print("------------------------------------------------------")
        return {"matches": matches}

    except Exception as e:
        print("❌ Pinecone query failed:", e)
        return {"matches": []}
