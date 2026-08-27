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
from pathlib import Path

import groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    extract_new_tool_calls,
    run_turn_with_memory,
    start_conversation,
    verify_and_start_conversation,
)
from businessflow.channels.credentials import looks_like_credentials, parse_credentials

app = FastAPI(title="BusinessFlow Chat API")

# The borrower-facing chat's static frontend -- mounted at the bottom of
# this file, after every API route, same reasoning as ops/api.py's own
# static mount: an exact-path route like POST /conversations is always
# matched first, so the mount only ever serves /static/styles.css etc.
_STATIC_DIR = Path(__file__).resolve().parent / "static"

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
    # Set only on the turn where an anonymous session just verified via
    # credentials typed into the chat (see send_message_endpoint) -- lets
    # the frontend update its own account-bound UI (the header chip)
    # without parsing the reply text to detect what just happened.
    verified_account_id: str | None = None


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
        "language": req.language,
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

    # An anonymous session can still verify mid-conversation by sending
    # "<account_id> <6-digit key>" as a plain message -- the same pattern
    # telegram_bot.py already handles (see channels/credentials.py). Found
    # live: a borrower typing their real account_id + access_key straight
    # into an anonymous browser chat had both values forwarded to the LLM
    # as free text, which then passed the ACCESS KEY as account_id to a
    # tool and crashed it (get_payment_status raised "no account found for
    # account_id='<the key>'"). Checked here, before this message ever
    # reaches run_turn_with_memory, exactly like Telegram's guard.
    if session["account_id"] is None and looks_like_credentials(req.message):
        account_id, access_key = parse_credentials(req.message)
        try:
            conversation = verify_and_start_conversation(session["language"], account_id, access_key)
        except AccessDeniedError:
            return SendMessageResponse(
                reply=f"That access key doesn't match account {account_id} -- please check and send both again.",
                tool_calls=[],
            )
        except AccountLockedError as e:
            return SendMessageResponse(reply=str(e), tool_calls=[])
        # Verification itself is not a turn -- run_turn_with_memory is
        # deliberately not called for this message, matching
        # telegram_bot.py's handle_incoming_message. The anonymous
        # session's history is discarded outright: its system prompt
        # never knew an account_id existed, so there's nothing in it
        # worth carrying into the newly-verified conversation.
        session["account_id"] = account_id
        session["messages"] = conversation
        return SendMessageResponse(
            reply=f"Verified -- I've pulled up account {account_id}. What can I help you with?",
            tool_calls=[],
            verified_account_id=account_id,
        )

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


@app.get("/", include_in_schema=False)
def chat_ui():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
