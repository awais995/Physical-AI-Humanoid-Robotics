from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from .api.chat_endpoints import router as chat_router
from .config.settings import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown events
    """
    # Startup
    logger.info("Starting up RAG Chatbot API...")
    try:
        # Any startup tasks can go here
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down RAG Chatbot API...")
        # Any cleanup tasks can go here


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG Chatbot API for the Physical AI & Humanoid Robotics Book",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router)

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API for Physical AI & Humanoid Robotics Book",
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "chat": "/chat/query",
            "conversations": "/chat/conversations",
            "ingest": "/chat/ingest",
            "health": "/chat/health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )