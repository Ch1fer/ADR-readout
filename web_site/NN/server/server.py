import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os


app = FastAPI()


def delete_files_in_directory(directory):
    dir_path = Path(directory)

    files = dir_path.glob('*')

    for file in files:
        file.unlink()


@app.post("/upload")
async def upload_image(image: UploadFile):
    directory = "client_files"

    delete_files_in_directory(directory)

    contents = await image.read()
    with open(os.path.join(directory, image.filename), "wb") as f:
        f.write(contents)

    return {"detail": "File uploaded successfully"}


origins = [
    "http://localhost:63342",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)