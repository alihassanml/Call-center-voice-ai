import asyncio
import livekit
from faster_whisper import WhisperModel

# Initialize Whisper ASR model
model = WhisperModel("small", device="cuda", compute_type="float16")

async def on_audio(track):
    async for frame in track:
        audio_data = frame.data  # Get raw audio
        segments, _ = model.transcribe(audio_data)
        for segment in segments:
            print("User said:", segment.text)  # Process transcription

async def main():
    room = await livekit.Room.connect("ws://your-livekit-server", "api_key")
    room.on("track_subscribed", on_audio)

    await room.join()
    print("Listening for calls...")

asyncio.run(main())
