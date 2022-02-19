from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define request schema
class PollModel(BaseModel):
    time_stamp: int = ""
    audio_file: str = ""
    mic_number: int = 0


# Define get request
@app.get("/poll")
def poll_audio(request: PollModel):
    # load
    pass


