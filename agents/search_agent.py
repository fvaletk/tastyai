import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

# Init OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

@tool
def search_recipes(preferences: dict) -> dict:
    """
    Search for matching recipes in Pinecone based on extracted user preferences.
    """

    # Build a descriptive query string
    query_terms = []

    cuisine = preferences.get("cuisine", "")
    diet = preferences.get("diet", "")

    if cuisine and cuisine != "unknown":
        query_terms.append(cuisine)
    if diet and diet != "unknown":
        query_terms.append(diet)

    query_text = " ".join(query_terms)
    if not query_text:
        query_text = "healthy dinner"

    print(f"🔍 Embedding query: {query_text}")

    try:
        # Get embedding for the query
        embed_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_vector = embed_resp.data[0].embedding

        # Search Pinecone
        pinecone_response = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )

        # Extract recipe results
        matches = []
        for match in pinecone_response.matches:
            meta = match.metadata

            # Prettify stringified lists
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

        return {"matches": matches}

    except Exception as e:
        print("❌ Pinecone query failed:", e)
        return {"matches": []}
