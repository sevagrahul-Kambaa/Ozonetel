import fastapi
from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import requests
import traceback
from pydantic import BaseModel, HttpUrl

# Import OpenAI library
from openai import OpenAI

# --- AUTHENTICATION SETUP ---
print("[INIT] Starting authentication setup...")
openai_api_key = "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"
client = OpenAI(api_key=openai_api_key)
print("SUCCESS: OpenAI client initialized")
print("[INIT] Authentication setup complete")
# --- END SETUP ---

app = fastapi.FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()},
    )

class AudioURLPayload(BaseModel):
    url: HttpUrl


@app.post("/transcribe")
def analyze_audio_from_url(payload: AudioURLPayload):
    print("\n" + "="*60)
    print(f"[REQUEST] Received request for URL: {payload.url}")
    print("="*60)
    supported_mime_types = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac", "audio/x-m4a", "audio/mp3"]
    
    try:
        # 1. Download Audio
        print("[STEP 1] Downloading audio file...")
        response = requests.get(str(payload.url), timeout=30)
        print(f"[STEP 1] Response status code: {response.status_code}")
        response.raise_for_status()

        audio_bytes = response.content
        print(f"[STEP 1] Downloaded {len(audio_bytes)} bytes")
        content_type = response.headers.get("Content-Type", "").lower()
        print(f"[STEP 1] Content-Type: {content_type}")

        # Basic validation (allow audio/mpeg even if extended params exist)
        print("[STEP 1] Validating content type...")
        if not any(t in content_type for t in supported_mime_types):
             # Fallback: if header is weird but it's an mp3 link, treat as audio/mpeg
            if str(payload.url).endswith(".mp3"):
                print("[STEP 1] Using fallback: treating as audio/mpeg")
                content_type = "audio/mpeg"
            else:
                print(f"[STEP 1] ERROR: Unsupported format: {content_type}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format: {content_type}. Supported: {supported_mime_types}",
                )
        print(f"[STEP 1] Content type validated: {content_type}")

        # 2. Transcribe with OpenAI Whisper
        print("[STEP 2] Starting Whisper transcription...")
        
        # Save audio to temporary file for OpenAI API
        import tempfile
        print("[STEP 2] Creating temporary audio file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        print(f"[STEP 2] Temp file created: {temp_audio_path}")
        
        try:
            # Transcribe using Whisper
            print("[STEP 2] Sending audio to Whisper API...")
            with open(temp_audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            print("[STEP 2] Whisper transcription complete")
            
            # Get the full transcript
            transcript_text = transcription.text
            print(f"[STEP 2] Transcript length: {len(transcript_text)} characters")
            print(f"[STEP 2] Transcript preview: {transcript_text[:100]}...")
            
            # 3. Analyze with GPT-4
            print("[STEP 3] Starting GPT-4 analysis...")
            
            analysis_prompt = f"""
            Analyze the following audio transcript and generate a single, valid JSON object.
            **Your entire response MUST be a raw JSON object, without any surrounding text, notes, or markdown formatting like ```json.**
            
            The JSON object must contain exactly these three top-level keys:
            1. `diarized_transcript`: A string containing the full, timestamped transcript with each speaker on a new line. The speaker labels MUST be "Speaker A" and "Speaker B". Infer speaker changes from context.
            2. `summary`: A string containing a concise summary of the conversation.
            3. `tone_analysis`: An object where each key is a speaker label (e.g., "Speaker A") and its corresponding value is a single string representing the speaker's dominant emotion. The value MUST be one of the following four options: "happy", "sad", "fear", or "angry" — followed by a short explanation of why that tone was inferred.
            
            Transcript:
            {transcript_text}
            """
            
            print("[STEP 3] Sending analysis request to GPT-4o...")
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing conversations and generating structured JSON output."},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format={"type": "json_object"}
            )
            print("[STEP 3] GPT-4o analysis complete")
            
            response_text = completion.choices[0].message.content
            print(f"[STEP 3] Response length: {len(response_text)} characters")
            
            # 4. Parse JSON
            print("[STEP 4] Parsing JSON response...")
            cleaned_text = response_text.strip().replace("```json", "").replace("```", "").strip()
            json_response = json.loads(cleaned_text)
            print("[STEP 4] JSON parsed successfully")
            print(f"[STEP 4] Response keys: {list(json_response.keys())}")
            
        finally:
            # Clean up temp file
            print(f"[CLEANUP] Removing temp file: {temp_audio_path}")
            os.unlink(temp_audio_path)
            print("[CLEANUP] Temp file removed")

        print("[SUCCESS] Request completed successfully")
        print("="*60 + "\n")
        return json_response

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request exception: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch audio: {e}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON Decode Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model returned invalid JSON"
        )
    except Exception as e:
        error_msg = str(e)
        print("[ERROR] Unexpected Error:")
        print(traceback.format_exc())
        print("="*60 + "\n")
        
        # Handle OpenAI-specific errors
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            print("[ERROR] Authentication error detected")
            raise HTTPException(status_code=500, detail="OpenAI Authentication Error - Check API Key")
        elif "rate_limit" in error_msg.lower():
            print("[ERROR] Rate limit error detected")
            raise HTTPException(status_code=429, detail="OpenAI Rate Limit Exceeded")
        else:
            print(f"[ERROR] Generic error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Internal error: {error_msg}")

@app.get("/")
def read_root():
    return {"message": "OpenAI Audio Analysis API is running"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
