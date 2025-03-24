from gtts import gTTS
import os

text = "Hello, how can I assist you?"
tts = gTTS(text=text, lang="en")
tts.save("output.mp3")

