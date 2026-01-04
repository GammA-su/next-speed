import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from infer import SentenceGenerator

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI()
GENERATOR = SentenceGenerator()


class GenerateRequest(BaseModel):
    prompt: str
    max_sentences: int = 8
    params: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_sentences: int = 8
    params: Optional[Dict[str, Any]] = None


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        content = msg.content.strip()
        if not content:
            continue
        parts.append(content)
    return " ".join(parts)


@app.post("/generate")
def generate(req: GenerateRequest):
    params = req.params or {}
    return GENERATOR.generate(req.prompt, req.max_sentences, **params)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    params = req.params or {}
    prompt = messages_to_prompt(req.messages)
    result = GENERATOR.generate(prompt, req.max_sentences, **params)
    assistant_text = " ".join(result["sentences"]).strip()

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "sentence-generator",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop",
            }
        ],
    }


def main() -> None:
    import uvicorn

    uvicorn.run("09_server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
