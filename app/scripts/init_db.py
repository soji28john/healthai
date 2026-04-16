import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.models.db_models import Base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./healthai.db")

async def init():
    engine = create_async_engine(DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created successfully.")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init())