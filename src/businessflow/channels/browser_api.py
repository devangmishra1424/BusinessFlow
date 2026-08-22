"""HTTP API for a browser-based text chat channel -- the backend a
frontend can call (built separately, per the plan: backend first, then
UI/UX). Text only for now, matching the project's "text primary, voice
later" decision.

Conversation state is held in-process, in memory, keyed by a generated
conversation_id -- restarting this server loses any conversation still
in flight. Cross-session memory (the recap a returning borrower's NEXT
conversation gets) is unaffected, since that's persisted to Postgres via
memory/conversation_memory.py independently of this in-memory state.

Run: uvicorn businessflow.channels.browser_api:app --reload
(defaults to port 8000). The ops dashboard API (ops/api.py) is a
separate FastAPI app on port 8001 -- run both side by side, they don't
share state or a port.
"""

import uuid

import groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    extract_new_tool_calls,
    run_turn_with_memory,
    start_conversation,
    verify_and_start_conversation,
)

app = FastAPI(title="BusinessFlow Chat API")

# Dev-only: lets a locally-served frontend on any port call this API
# without a CORS error. Tighten to a specific origin before any real
# deployment -- this is intentionally permissive for local frontend work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_conversations: dict[str, dict] = {}


class StartConversationRequest(BaseModel):
    account_id: str | None = None
    access_key: str | None = None  # required if account_id is given
    language: str = "en"


class StartConversationResponse(BaseModel):
    conversation_id: str
    account_id: str | None
    language: str


class SendMessageRequest(BaseModel):
    message: str


class ToolCallInfo(BaseModel):
    tool: str
    arguments: dict


class SendMessageResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallInfo]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/conversations", response_model=StartConversationResponse)
def start_conversation_endpoint(req: StartConversationRequest):
    if req.language not in ("en", "hi"):
        raise HTTPException(status_code=400, detail="language must be 'en' or 'hi'")

    if req.account_id:
        if not req.access_key:
            raise HTTPException(status_code=401, detail=f"access_key is required to talk about account {req.account_id}")
        try:
            conversation = verify_and_start_conversation(req.language, req.account_id, req.access_key)
        except AccessDeniedError:
            raise HTTPException(status_code=401, detail=f"wrong access key for account {req.account_id}") from None
        except AccountLockedError as e:
            raise HTTPException(status_code=429, detail=str(e)) from None
    else:
        conversation = start_conversation(language=req.language, account_id=None)

    conversation_id = str(uuid.uuid4())
    _conversations[conversation_id] = {
        "account_id": req.account_id,
        "messages": conversation,
    }
    return StartConversationResponse(conversation_id=conversation_id, account_id=req.account_id, language=req.language)


@app.post("/conversations/{conversation_id}/messages", response_model=SendMessageResponse)
def send_message_endpoint(conversation_id: str, req: SendMessageRequest):
    session = _conversations.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no conversation found for id={conversation_id!r}")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    turn_start = len(session["messages"])
    session["messages"].append({"role": "user", "content": req.message})
    try:
        updated_conversation, reply = run_turn_with_memory(session["messages"], session["account_id"])
    except groq.RateLimitError as e:
        # Something the frontend can actually act on (show "try again in a
        # bit", maybe auto-retry) -- not the same as a real bug, so it
        # shouldn't come back as an opaque 500.
        session["messages"].pop()  # don't leave a user message with no reply appended
        raise HTTPException(status_code=503, detail=f"Groq rate limit reached: {e.message}") from e
    except groq.APIStatusError as e:
        session["messages"].pop()
        raise HTTPException(status_code=502, detail=f"upstream LLM provider error: {e.message}") from e
    session["messages"] = updated_conversation

    tool_calls = extract_new_tool_calls(updated_conversation, turn_start)
    return SendMessageResponse(
        reply=reply,
        tool_calls=[ToolCallInfo(tool=name, arguments=args) for name, args in tool_calls],
    )
