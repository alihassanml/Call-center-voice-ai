import pyttsx3

tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')

# Change to a specific voice (e.g., index 1)
tts_engine.setProperty('voice', voices[1].id)
tts_engine.setProperty('rate',140)
tts_engine.say("Hello, this is a different voice!")
tts_engine.runAndWait()
