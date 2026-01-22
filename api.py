#!/usr/bin/env python3
"""
OpenAI-compatible API server for self-notes RAG system.
Allows Open WebUI and other OpenAI-compatible clients to use the RAG pipeline.
"""
import time
import uuid
import warnings
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Suppress pydantic v1 compatibility warning (langchain internal)
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

from src.query import query
from src.config import LLM_MODEL

app = FastAPI(title="Self-Notes RAG API", version="1.0.0")

# Allow CORS for Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str
    data: list[ModelInfo]


@app.get("/v1/models", response_model=ModelsResponse)
@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id="self-notes",
                object="model",
                created=int(time.time()),
                owned_by="local"
            )
        ]
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions using the RAG pipeline."""

    # Extract the last user message as the question
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        question = ""
    else:
        question = user_messages[-1].content

    # Query the RAG system
    result = query(question)
    answer = result["answer"]

    # Append sources if available
    if result["sources"]:
        answer += "\n\n---\n**Sources:**\n"
        for src in result["sources"]:
            answer += f"- [{src['title']}]({src['url']})\n"

    # Handle streaming
    if request.stream:
        import json

        async def generate():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())

            # Send role chunk first
            role_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "self-notes",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            # Stream content in smaller chunks
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                content_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "self-notes",
                    "choices": [{"index": 0, "delta": {"content": answer[i:i+chunk_size]}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

            # Send done chunk
            done_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "self-notes",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming response
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model="self-notes",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=len(question.split()),
            completion_tokens=len(answer.split()),
            total_tokens=len(question.split()) + len(answer.split())
        )
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
