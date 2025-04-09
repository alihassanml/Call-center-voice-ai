from faster_whisper import WhisperModel
import pyaudio
import numpy as np

# Initialize Whisper model (tiny for speed)
model = WhisperModel("tiny.en")

# Initialize PyAudio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
stream.start_stream()

print("🎤 Listening...")

while True:
    audio_data = stream.read(1024, exception_on_overflow=False)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Transcribe the audio chunk
    segments, _ = model.transcribe(audio_np, language="en", beam_size=1)

    if segments:
        for segment in segments:
            print(f"\r🗣 {segment.text}", end="")
