from fastapi import FastAPI
from web_site.NN.server.src.endpoints import router
from starlette.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def start_server():
    uvicorn.run(app, host="localhost", port=7777)


if __name__ == "__main__":
    start_server()
