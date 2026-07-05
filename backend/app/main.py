import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.api.routes import documents, problems, hypotheses, search, graph, export
from app.db.database import init_db
from app.db.qdrant_client import init_qdrant
from app.services.glm_health import check_glm_health
from app.services.paddle_health import check_paddle_health
from app.services.processing_queue import get_processing_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_qdrant()
    await check_glm_health()
    check_paddle_health()
    queue = get_processing_queue()
    await queue.start()
    yield
    await queue.stop()


app = FastAPI(title="Hypothesis Factory", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(problems.router, prefix="/api/v1/problems", tags=["problems"])
app.include_router(hypotheses.router, prefix="/api/v1/hypotheses", tags=["hypotheses"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(export.router, prefix="/api/v1/export", tags=["export"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="web-ui", html=True), name="web-ui")
