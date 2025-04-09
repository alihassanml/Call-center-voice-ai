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
from gtts import gTTS
import pygame
import io


load_dotenv()


index = faiss.read_index("./code/call_center_faiss.index")
with open("./code/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


df = pd.read_csv("./code/data.csv")
assistant_responses = df[df["role"] == "assistant"]["content"].tolist()


client = OpenAI(api_key=os.getenv("OPEN_AI"))


recognizer = sr.Recognizer()
stop_speech_event = threading.Event()
listening_active = True  


try:
    tts_engine = pyttsx3.init(driverName='espeak')
except Exception:
    tts_engine = pyttsx3.init()


pygame.mixer.init()

def find_best_response(user_query):
    query_vector = vectorizer.transform([user_query]).toarray()
    _, idx = index.search(query_vector, 1)
    return assistant_responses[idx[0][0]]

def play_audio(text):
    """Plays generated audio response using gTTS."""
    tts = gTTS(text, lang="en")
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)

    pygame.mixer.music.stop()
    
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()


def get_gpt_response(user_query, retrieved_response):
    """Generates an AI response using OpenAI."""
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
    print(f" AI Response: {response_text}")
    
    play_audio(response_text)

def listen():
    """Continuously listens and processes speech input."""
    global listening_active

    while listening_active:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)

            try:
                stop_speech_event.set()
                audio = recognizer.listen(source, timeout=5)
                print("Processing speech...")

                user_query = recognizer.recognize_google(audio)
                stop_speech_event.clear()

                if user_query:
                    print(f"User: {user_query}")
                    retrieved_response = find_best_response(user_query)
                    get_gpt_response(user_query, retrieved_response)
            except sr.UnknownValueError:
                print("Could not understand audio")
                stop_speech_event.clear()
            except sr.RequestError:
                print("Could not reach speech recognition service")
                stop_speech_event.clear()
            except sr.WaitTimeoutError:
                print("Listening timed out, retrying...")
                stop_speech_event.clear()

def main():
    threading.Thread(target=listen, daemon=True).start()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
