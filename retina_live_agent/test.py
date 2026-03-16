"""
Quick test — checks if GOOGLE_API_KEY in .env is valid for Gemini API.
Run: python test.py
"""

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent / ".env")

api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key:
    print("FAIL — GOOGLE_API_KEY is missing in .env")
    exit(1)

print(f"Key found: {api_key[:8]}...{api_key[-4:]}")

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Say: KEY_VALID"
    )
    print("PASS — API key is valid")
    print("Response:", response.text.strip())
except Exception as e:
    print("FAIL — API key is invalid")
    print("Error:", e)
