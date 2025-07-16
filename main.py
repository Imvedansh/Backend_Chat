from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import google.generativeai as genai

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use Gemini model
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# FastAPI app setup
app = FastAPI()

# Allow frontend (React) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001",   "https://bot-chat-frontend-iyj5.vercel.app", ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== ROUTE 1: NORMAL CHAT with STREAMING ==== #
class Prompt(BaseModel):
    message: str

@app.post("/chat")
async def chat_api(prompt: Prompt):
    def event_stream():
        try:
            response = model.generate_content(prompt.message, stream=True)
            for chunk in response:
                yield chunk.text
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    return StreamingResponse(event_stream(), media_type="text/plain")

# ==== ROUTE 2: UPLOAD PDF ==== #
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("temp.pdf", "wb") as f:
            f.write(contents)

        # Extract text using PyMuPDF
        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        return {"text": text[:5000]}  # Limit to 5K characters
    except Exception as e:
        return {"text": f"❌ Failed to read PDF: {str(e)}"}

# ==== ROUTE 3: ASK QUESTION ABOUT PDF with STREAMING ==== #
class PDFPrompt(BaseModel):
    message: str
    context: str

@app.post("/ask-pdf")
async def ask_about_pdf(prompt: PDFPrompt):
    def pdf_event_stream():
        try:
            full_prompt = (
                f"You are reading the following document:\n\n{prompt.context}\n\n"
                f"Based on this, answer the question:\n{prompt.message}"
            )
            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                yield chunk.text
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    return StreamingResponse(pdf_event_stream(), media_type="text/plain")
