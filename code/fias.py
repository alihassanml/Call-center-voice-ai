import os
import faiss
import openai
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import pickle


load_dotenv()

df = pd.read_csv("data/data/data.csv")

user_queries = df[df["role"] == "user"]["content"].tolist()
assistant_responses = df[df["role"] == "assistant"]["content"].tolist()

vectorizer = TfidfVectorizer()
query_vectors = vectorizer.fit_transform(user_queries).toarray()
pickle.dump(vectorizer, open("./code/vectorizer.pkl", "wb"))

dimension = query_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(query_vectors)

faiss.write_index(index, "./code/call_center_faiss.index")

def find_best_response(user_query):
    query_vector = vectorizer.transform([user_query]).toarray()
    _, idx = index.search(query_vector, 1)
    return assistant_responses[idx[0][0]]

client = OpenAI(api_key=os.getenv("OPEN_AI"))

def get_gpt_response(user_query, retrieved_response):
    prompt = f"""
    The customer said: {user_query}
    The best matching predefined response is: {retrieved_response}
    You are a helpful AI for a call center.
    If the predefined response fits, return it as is. If needed, improve it for better engagement.
    Give me just response to the user query.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

query = "I'm busy right now."
retrieved_response = find_best_response(query)
final_response = get_gpt_response(query, retrieved_response)

print("Final AI Response:", final_response)
