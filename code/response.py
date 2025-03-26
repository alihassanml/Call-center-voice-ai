import os
import time
import faiss
import openai
import pickle
import threading
import queue
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

index = faiss.read_index("./code/call_center_faiss.index")
with open("./code/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

df = pd.read_csv("data/data/data.csv")
assistant_responses = df[df["role"] == "assistant"]["content"].tolist()


client = OpenAI(api_key=os.getenv("OPEN_AI"))
recognizer = sr.Recognizer()
stop_speech_queue = queue.Queue()

tts_engine = pyttsx3.init(driverName='espeak')


def find_best_response(user_query):
    query_vector = vectorizer.transform([user_query]).toarray()
    _, idx = index.search(query_vector, 1)
    return assistant_responses[idx[0][0]]

def play_audio(text):
    global tts_engine
    tts_engine.say(text)
    tts_engine.runAndWait()

def get_gpt_response(user_query, retrieved_response):
    prompt = f"""
    The customer said: {user_query}
    The best matching predefined response is: {retrieved_response}
    You are a helpful AI for a call center.
    If the predefined response fits, return it as is. If needed, improve it for better engagement.
    Give me just response to the usretrieved_responseer query.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = response.choices[0].message.content
    play_audio(response_text)



def listen():
    """Listen to user input and return transcribed text."""
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        
        try:
            audio = recognizer.listen(source, timeout=5)
            print("📝 Processing speech...")
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
        except sr.WaitTimeoutError:
            return None

def main():
    """Main interactive loop."""
    while True:
        user_query = listen()
        
        if user_query:
            print(f"🗣 User: {user_query}")
            
            stop_speech_queue.put(True)
            
            retrieved_response = find_best_response(user_query)
            get_gpt_response(user_query, retrieved_response)
            
            

if __name__ == "__main__":
    main()
