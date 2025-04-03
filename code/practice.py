import os
import time
import faiss
import openai
import pickle
import threading
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

index = faiss.read_index("./call_center_faiss.index")
with open("./vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

df = pd.read_csv("./data.csv")
assistant_responses = df[df["role"] == "assistant"]["content"].tolist()

client = OpenAI(api_key=os.getenv("OPEN_AI"))

recognizer = sr.Recognizer()
stop_speech_event = threading.Event()
listening_active = True  

tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[1].id)
tts_engine.setProperty('rate', 150)

def find_best_response(user_query):
    query_vector = vectorizer.transform([user_query]).toarray()
    _, idx = index.search(query_vector, 1)
    return assistant_responses[idx[0][0]]

tts_lock = threading.Lock()

def play_audio(text):
    with tts_lock:
        tts_engine.say(text)
        tts_engine.runAndWait()

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
    response_text = response.choices[0].message.content
    print(f"🤖 AI Response: {response_text}")
    
    play_audio(response_text)

def listen():
    global listening_active

    while listening_active:
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source)

            try:
                stop_speech_event.set()
                audio = recognizer.listen(source, timeout=3)
                print("📝 Processing speech...")

                user_query = recognizer.recognize_google(audio)
                stop_speech_event.clear()

                if user_query:
                    print(f"🗣 User: {user_query}")
                    retrieved_response = find_best_response(user_query)
                    get_gpt_response(user_query, retrieved_response)

            except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError) as e:
                print(f"⚠ Error: {e}")
                stop_speech_event.clear()
                continue

def interrupt_speech():
   
    global listening_active

    while listening_active:
        if tts_engine._inLoop:
            print("Interrupting speech...")
            tts_engine.stop()  
            time.sleep(0.1)  

def main():
    threading.Thread(target=listen, daemon=True).start()
    threading.Thread(target=interrupt_speech, daemon=True).start()  
    while True:
        time.sleep(0.01)

if __name__ == "__main__":
    main()
