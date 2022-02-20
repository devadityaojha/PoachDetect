from fastapi import FastAPI, Depends
from starlette.requests import Request
import uvicorn

from app.api.api_v1.routers.users import users_router
from app.api.api_v1.routers.auth import auth_router
from app.core import config
from app.db.session import SessionLocal
from app.core.auth import get_current_active_user
from app.core.celery_app import celery_app
from app import tasks
from mage_classifier import post_request
from classifier import classify_audio

NORMALIZATION_FACTOR = 0.0001
MAGE_WEIGHT = 0.05
AUDIO_WEIGHT = 0.95

app = FastAPI(
    title=config.PROJECT_NAME, docs_url="/api/docs", openapi_url="/api"
)


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    request.state.db = SessionLocal()
    response = await call_next(request)
    request.state.db.close()
    return response


@app.get("/api/v1")
async def root():
    return {"message": "Hello World"}


@app.get("/api/v1/task")
async def example_task():
    celery_app.send_task("app.tasks.example_task", args=["Hello World"])

    return {"message": "success"}

@app.get("/api/v1/task/poacher_search")
async def poacher_search():
    success_probability = post_request()[0]
    location = post_request()[1][0]
    coordinates = post_request()[1][1]
    audio_classification = classify_audio()
    val = NORMALIZATION_FACTOR * MAGE_WEIGHT * success_probability + AUDIO_WEIGHT * audio_classification

    return {"message": {
        "Payload": "Detected" if val >= 0.5 else "Not Detected",
        "Location": location,
        "Coordinates": coordinates
    }


# Routers
app.include_router(
    users_router,
    prefix="/api/v1",
    tags=["users"],
    dependencies=[Depends(get_current_active_user)],
)
app.include_router(auth_router, prefix="/api", tags=["auth"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8888)
