Audio Transcription and Summarization with FastAPI

This project is a FastAPI-based system that handles audio files by transcribing them, summarizing the content, extracting timestamps, and saving the results locally. It utilizes the Whisper model from OpenAI for transcription and the Hugging Face transformers library for summarization.

Features

Audio Transcription: Uses OpenAI's Whisper model to transcribe audio files.

Summarization: Summarizes the transcribed text using a Hugging Face summarization model.

Timestamp Extraction: Extracts timestamps from the transcription.

File Handling: Saves transcription, summary, and timestamps locally.

Requirements

Python 3.8+
FastAPI
Uvicorn
aiofiles
OpenAI API key
transformers
torch
