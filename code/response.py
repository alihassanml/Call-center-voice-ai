import os
import faiss
import openai
import pandas as pd
import time
import pickle
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load FAISS index and vectorizer
index = faiss.read_index("call_center_faiss.index")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load CSV data
df = pd.read_csv("data/data/data.csv")
user_queries = df[df["role"] == "user"]["content"].tolist()
assistant_responses = df[df["role"] == "assistant"]["content"].tolist()

def find_best_response(user_query):
    query_vector = vectorizer.transform([user_query]).toarray()
    _, idx = index.search(query_vector, 1)
    return assistant_responses[idx[0][0]]

client = OpenAI(api_key=os.getenv("OPEN_AI"))

def get_gpt_response(user_query, retrieved_response):
    prompt = f"""
    The customer said: {user_query}
    The best matching predefined response is: {retrieved_response}
    
    If the predefined response fits, return it as is. If needed, improve it for better engagement.
    """

    start_time = time.time()  # Start measuring time
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    end_time = time.time()  # End measuring time
    
    response_time = end_time - start_time  # Calculate response time
    return response.choices[0].message.content, response_time

# Test query
query = "I need to think about it."
retrieved_response = find_best_response(query)

final_response, response_time = get_gpt_response(query, retrieved_response)

print("Final AI Response:", final_response)
print(f"Response Time: {response_time:.2f} seconds")
