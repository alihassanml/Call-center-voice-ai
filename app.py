import openai
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPEN_AI")


openai.api_key = os.getenv("OPEN_AI")

response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "You are a helpful AI for a call center."},
              {"role": "user", "content": "Hello, how can you assist in a call center?"}]
)

print(response["choices"][0]["message"]["content"])
