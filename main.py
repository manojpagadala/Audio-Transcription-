from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import os
import uuid
import openai
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

UPLOAD_DIR = "./uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

openai.api_key = "API key"  

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File("Maruvaali.mp3")):
    try:
        logger.info(f"Received file: {file.filename} with content type: {file.content_type}")
        if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3"]:
            logger.error("Invalid audio format")
            raise HTTPException(status_code=400, detail="Invalid audio format")
        
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        logger.info(f"File saved to: {file_path}")
        return {"file_id": file_id, "file_path": file_path}
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

async def transcribe_audio(file_path: str) -> str:
    try:
        logger.info(f"Transcribing audio file: {file_path}")
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", file=audio_file)
            logger.info("Transcription successful")
            return transcript['text']
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

async def summarize_text(text: str) -> str:
    try:
        logger.info("Summarizing text")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following text:\n\n{text}",
            max_tokens=150
        )
        summary = response.choices[0].text.strip()
        logger.info("Summarization successful")
        return summary
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

async def extract_timestamps(file_path: str):
    try:
        logger.info(f"Extracting timestamps from audio file: {file_path}")
        audio = AudioSegment.from_file(file_path)
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
        timestamps = [{"start": start / 1000, "end": end / 1000} for start, end in nonsilent_ranges]
        logger.info("Timestamp extraction successful")
        return timestamps
    except Exception as e:
        logger.error(f"Timestamp extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Timestamp extraction failed: {str(e)}")

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        
        logger.info("Process audio file upload")
        upload_response = await upload_audio(file)
        file_path = upload_response["file_path"]
        
        transcription = await transcribe_audio(file_path)
        
        # Summarize transcription
        summary = await summarize_text(transcription)
        #  timestamps
        timestamps = await extract_timestamps(file_path)
        
        
        result = {
            "transcription": transcription,
            "summary": summary,
            "timestamps": timestamps
        }
        
        result_file = os.path.join(UPLOAD_DIR, f"{upload_response['file_id']}_result.json")
        async with aiofiles.open(result_file, 'w') as f:
            await f.write(json.dumps(result, indent=2))
        
        logger.info("Process completed successfully")
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)
