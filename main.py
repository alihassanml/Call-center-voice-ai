import os
from openai import OpenAI
from time import sleep
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key = os.getenv("OPEN_AI"))


def test_model(test_input):
    completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:cmp::BEFBTwzx",
        messages=[
            {
                "role": "system",
                "content": "I already have insurance."
            },
            {"role": "user", "content": test_input}
        ]
    )
    return completion.choices[0].message

test_report = "I need to talk to my spouse first"

# Get prediction
result = test_model(test_report)
print(f"Prediction: {result.content}")