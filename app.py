import openai

OPENAI_API_KEY = "your-api-key-here"

openai.api_key = OPENAI_API_KEY

response = openai.ChatCompletion.create(
    model="gpt-4-turbo",
    messages=[{"role": "system", "content": "You are a helpful AI for a call center."},
              {"role": "user", "content": "Hello, how can you assist in a call center?"}]
)

print(response["choices"][0]["message"]["content"])
