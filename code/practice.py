import os
import time
import faiss
import openai
import pickle
import pandas as pd
import speech_recognition as sr
import pyttsx3
from faster_whisper import WhisperModel
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load FAISS index and vectorizer
index = faiss.read_index("./call_center_faiss.index")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load dataset
df = pd.read_csv("data/data/data.csv")
assistant_responses = df[df["role"] == "assistant"]["content"].tolist()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPEN_AI"))

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # Adjust speed

# Load Whisper Model (Use int8 for CPU)
model = WhisperModel("small", device="cpu", compute_type="int8")

# Function to find best response using FAISS
def find_best_response(user_query):
    query_vector = vectorizer.transform([user_query]).toarray()
    _, idx = index.search(query_vector, 1)
    return assistant_responses[idx[0][0]]

# Function to get ChatGPT response
def get_gpt_response(user_query):
    retrieved_response = find_best_response(user_query)

    prompt = f"Customer: {user_query}\nBest response: {retrieved_response}\nRefine if needed."

    start_time = time.time()  # Start timing

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    end_time = time.time()  # Stop timing
    response_time = end_time - start_time

    return response.choices[0].message.content, response_time

# Function to convert text to speech
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to listen and transcribe voice
def listen_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening... Speak now.")
        audio = recognizer.listen(source)

        # Save audio to file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())

        print("📝 Transcribing...")
        segments, _ = model.transcribe("temp_audio.wav")
        transcribed_text = "".join([segment.text for segment in segments])

        print(f"✅ Transcribed: {transcribed_text}")
        return transcribed_text

# Main loop
while True:
    try:
        user_query = listen_and_transcribe()

        if user_query.lower() in ["exit", "quit", "stop"]:
            print("🛑 Exiting...")
            break

        final_response, response_time = get_gpt_response(user_query)

        print(f"🤖 AI Response: {final_response}")
        print(f"⚡ Response Time: {response_time:.2f} seconds")

        speak(final_response)

    except Exception as e:
        print(f"❌ Error: {e}")
