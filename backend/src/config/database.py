import os
from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL", "")

# For async operations with Neon Postgres - using asyncpg driver
async_engine = create_async_engine(
    NEON_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,  # Set to True for SQL query logging
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=300,
)

# Synchronous engine for any sync operations if needed
sync_engine = create_engine(
    NEON_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=300,
)

# Async session maker
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Sync session maker
SyncSessionLocal = sessionmaker(
    sync_engine, expire_on_commit=False
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get async database session
    """
    async with AsyncSessionLocal() as session:
        yield session


def get_sync_db():
    """
    Dependency to get sync database session
    """
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize the database connection
async def init_db():
    """
    Initialize the database connection and create tables if they don't exist
    """
    # Import models here to register them with SQLAlchemy
    from ..models.conversation import Conversation
    from ..models.query import Query
    from ..models.response import Response

    # Create tables
    async with async_engine.begin() as conn:
        # Create tables - in a real app you'd use Alembic for migrations
        pass