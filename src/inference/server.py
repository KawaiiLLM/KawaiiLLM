"""FastAPI server for KawaiiLLM inference with SSE streaming."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .engine import KawaiiInferenceEngine

logger = logging.getLogger(__name__)

engine: KawaiiInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "output/kawaii_v1")
    device = os.environ.get("DEVICE", "cuda")
    attn_impl = os.environ.get("ATTN_IMPL", None)
    num_mem = int(os.environ.get("NUM_MEM_TOKENS", "128"))

    logger.info("Loading model from %s on %s ...", checkpoint_dir, device)
    engine = KawaiiInferenceEngine(
        checkpoint_dir=checkpoint_dir,
        num_mem_tokens=num_mem,
        device=device,
        attn_implementation=attn_impl,
    )
    logger.info("Model ready.")
    yield
    engine = None


app = FastAPI(title="KawaiiLLM Inference", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/api/memory")
async def set_memory(request: Request):
    if engine is None:
        return {"error": "Model not loaded"}, 503
    body = await request.json()
    memory_text = body.get("memory_text", "")
    n_mem = body.get("n_mem", None)
    engine.set_memory(memory_text, n_mem=n_mem)
    return {"status": "ok", "n_mem": engine._active_n_mem}


@app.post("/api/chat")
async def chat(request: Request):
    if engine is None:
        return {"error": "Model not loaded"}, 503
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return {"error": "Empty messages"}, 400
    params = body.get("params", {})

    async def event_stream():
        loop = asyncio.get_event_loop()
        # engine.generate(stream=True) returns a TextIteratorStreamer immediately.
        # The actual generation runs in a daemon thread inside generate().
        streamer = await loop.run_in_executor(
            None,
            lambda: engine.generate(messages, stream=True, **params),
        )
        # C3 fix: TextIteratorStreamer.__next__ blocks (queue.get), so we must
        # call it in a thread executor to avoid blocking the asyncio event loop.
        _sentinel = object()
        while True:
            token_text = await loop.run_in_executor(
                None, lambda: next(streamer, _sentinel)
            )
            if token_text is _sentinel:
                break
            data = json.dumps({"token": token_text}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
