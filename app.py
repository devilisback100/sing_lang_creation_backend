from fastapi import FastAPI, HTTPException
import requests
import imageio
import numpy as np
from PIL import Image
import base64
import io
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieve environment variables with default values if not set
GIF_BASE_URL = os.getenv("GIF_BASE_URL")
TRANSLATE_URL = os.getenv("TRANSLATE_URL")


def resize_frame(frame, target_height=480):
    img = Image.fromarray(frame)
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)
    resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    return np.array(resized)


def get_gif_frames(word):
    """
    Fetch GIF frames for the given word.
    Returns a tuple (frames, durations), where:
      - frames is a list of base64-encoded JPEG strings,
      - durations is a list of per-frame durations in seconds.
    """
    frames = []
    durations = []
    try:
        url = f"{GIF_BASE_URL}/{word}.gif"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Wrap response.content in BytesIO for imageio
            gif_reader = imageio.get_reader(
                io.BytesIO(response.content), '.gif')
            meta_data = gif_reader.get_meta_data()
            # Duration per frame in seconds (default to 0.1 sec if not provided)
            frame_duration = meta_data.get("duration", 100) / 1000.0
            for frame in gif_reader:
                resized_frame = resize_frame(frame)
                buffered = io.BytesIO()
                Image.fromarray(resized_frame).save(buffered, format="JPEG")
                frames.append(base64.b64encode(
                    buffered.getvalue()).decode("utf-8"))
                durations.append(frame_duration)
    except Exception as e:
        print(f"Error fetching GIF for '{word}': {e}")
    return frames, durations


@app.post("/get_frames")
async def get_frames_endpoint(request_data: dict):
    original_text = request_data.get("text", "")
    if not original_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Call the translation API to convert the original text into sign grammar
    translate_response = requests.post(
        TRANSLATE_URL, json={"text": original_text})
    if translate_response.status_code != 200:
        raise HTTPException(
            status_code=translate_response.status_code, detail="Translation API failed")
    translate_data = translate_response.json()
    sign_grammar = translate_data.get("sign_grammar", "")
    if not sign_grammar:
        raise HTTPException(status_code=400, detail="No sign grammar returned")

    # Convert sign grammar to lowercase so that GIF filenames match
    words = sign_grammar.lower().split()
    all_frames = []
    total_duration = 0.0  # Total duration in seconds

    for word in words:
        frames, durations = get_gif_frames(word)
        # If no frames found for the word, try processing letter by letter.
        if not frames:
            word_frames = []
            word_durations = []
            for letter in word:
                letter_frames, letter_durations = get_gif_frames(letter)
                # Only add letters that returned valid frames
                if letter_frames:
                    word_frames.extend(letter_frames)
                    word_durations.extend(letter_durations)
            frames = word_frames
            durations = word_durations

        total_duration += sum(durations)
        all_frames.append({
            "word": word,
            "frames": frames,
            "durations": durations
        })

    return {
        "original_text": original_text,
        "sign_grammar": sign_grammar,
        "frames": all_frames,
        "total_duration": total_duration
    }

# For local testing:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
