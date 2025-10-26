from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import traceback
import io
import os

from phq9_session import PHQ9Session

load_dotenv()
app = FastAPI()
client = OpenAI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/upload.html")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, file.file, file.content_type)
        )
        return {"transcript": result.text}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})

    try:
        response = client.audio.speech.create(
            model="tts-1",    # Replace with your actual TTS model
            voice="alloy",    # Replace with your preferred voice
            input=text
        )
        # Extract raw bytes from HttpxBinaryResponseContent
        audio_bytes = response.read()  # or response.content

        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# PHQ9 session management
session = PHQ9Session()


class PHQRequest(BaseModel):
    user_response: str


@app.post("/phq")
async def phq(data: PHQRequest):
    if data.user_response.strip().lower() == "start":
        return session.start()

    try:
        return session.process_response(data.user_response)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/phq/reset")
async def reset():
    global session
    session = PHQ9Session()
    return {"message": "Session reset."}

