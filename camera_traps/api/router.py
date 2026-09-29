from fastapi import APIRouter

from camera_traps.api.routes import home, image, video

api_router = APIRouter()

api_router.include_router(home.router, tags=["Home"])
api_router.include_router(image.router, tags=["Image"])
api_router.include_router(video.router, tags=["Video"])
