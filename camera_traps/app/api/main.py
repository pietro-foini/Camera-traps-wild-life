import logging

from fastapi import FastAPI

from camera_traps.app.api.routes import image_router, video_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="camera-traps-wildlife")

# Include API routes.
app.include_router(image_router)
app.include_router(video_router)
