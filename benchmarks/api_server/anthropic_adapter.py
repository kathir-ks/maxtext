"""Anthropic /v1/messages adapter for the MaxText API server.

This is a thin format translator. It does not implement any new
inference — it converts an Anthropic-shaped request into the internal
ChatCompletionRequest, forwards it through the same queue/batch path
that the OpenAI endpoints use, and rewrites the response into
Anthropic's shape so Claude Code (and the official `anthropic` Python
SDK) work without any client-side changes beyond setting
`ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`.

v1 limitations (intentional — out of scope per plan):
  - stream=True is rejected with 400.
  - tools / tool_choice / tool_use / tool_result blocks are rejected.
  - image content blocks are rejected (model is not multimodal).

Auth dependency is attached by the parent app via include_router(
    anthropic_router, dependencies=[Depends(require_api_key)]).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Union

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from benchmarks.api_server.server_models import ChatCompletionRequest, ChatMessage


router = APIRouter(tags=["anthropic"])


# ---------- Request models ----------------------------------------------------


class _TextBlock(BaseModel):
  type: Literal["text"]
  text: str


class _UnsupportedBlock(BaseModel):
  type: str

  class Config:
    extra = "allow"


ContentBlock = Union[_TextBlock, _UnsupportedBlock]


class AnthropicMessage(BaseModel):
  role: Literal["user", "assistant"]
  content: Union[str, list[ContentBlock]]


class AnthropicSystemBlock(BaseModel):
  type: Literal["text"]
  text: str


class AnthropicMessagesRequest(BaseModel):
  model: str
  messages: list[AnthropicMessage]
  system: Union[str, list[AnthropicSystemBlock], None] = None
  max_tokens: int = Field(..., gt=0)
  temperature: float | None = None
  top_p: float | None = None
  top_k: int | None = None
  stop_sequences: list[str] | None = None
  stream: bool = False
  metadata: dict[str, Any] | None = None
  # Fields we silently ignore / will support later. Pydantic allows
  # extras by default unless we opt in to strictness.

  class Config:
    extra = "allow"


# ---------- Response models ---------------------------------------------------


class AnthropicUsage(BaseModel):
  input_tokens: int
  output_tokens: int


class AnthropicTextContent(BaseModel):
  type: Literal["text"] = "text"
  text: str


class AnthropicMessageResponse(BaseModel):
  id: str
  type: Literal["message"] = "message"
  role: Literal["assistant"] = "assistant"
  model: str
  content: list[AnthropicTextContent]
  stop_reason: str | None
  stop_sequence: str | None = None
  usage: AnthropicUsage


# ---------- Conversion helpers ------------------------------------------------


def _flatten_content(content: Union[str, list[ContentBlock]]) -> str:
  if isinstance(content, str):
    return content
  parts: list[str] = []
  for block in content:
    bdict = block.model_dump() if hasattr(block, "model_dump") else dict(block)
    btype = bdict.get("type")
    if btype == "text":
      parts.append(str(bdict.get("text", "")))
    elif btype in {"tool_use", "tool_result", "image"}:
      raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail=f"content block type '{btype}' is not supported in v1",
      )
    else:
      raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail=f"unknown content block type '{btype}'",
      )
  return "".join(parts)


def _flatten_system(system: Union[str, list[AnthropicSystemBlock], None]) -> str | None:
  if system is None:
    return None
  if isinstance(system, str):
    return system
  return "".join(blk.text for blk in system if blk.type == "text")


_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "stop_sequence": "stop_sequence",
    # We don't currently use these but map defensively.
    "content_filter": "end_turn",
    "tool_calls": "tool_use",
}


def _to_anthropic_stop_reason(internal: str | None) -> str | None:
  if internal is None:
    return None
  return _FINISH_REASON_MAP.get(internal, "end_turn")


def _build_internal_request(req: AnthropicMessagesRequest) -> ChatCompletionRequest:
  if req.stream:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="stream=true is not supported in v1; use stream=false",
    )
  for extra in ("tools", "tool_choice"):
    if extra in (req.model_extra or {}):
      raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail=f"'{extra}' is not supported in v1",
      )

  messages: list[ChatMessage] = []
  system_text = _flatten_system(req.system)
  if system_text:
    messages.append(ChatMessage(role="system", content=system_text))
  for m in req.messages:
    messages.append(ChatMessage(role=m.role, content=_flatten_content(m.content)))

  return ChatCompletionRequest(
      model=req.model,
      messages=messages,
      max_tokens=req.max_tokens,
      temperature=req.temperature,
      top_p=req.top_p,
      top_k=req.top_k,
      stop=req.stop_sequences,
      stream=False,
      logprobs=False,
  )


# ---------- Endpoint ----------------------------------------------------------


@router.post("/v1/messages", response_model=AnthropicMessageResponse)
async def create_message(req: AnthropicMessagesRequest) -> AnthropicMessageResponse:
  internal_req = _build_internal_request(req)

  # Lazy import to avoid circular import at module load time
  # (maxtext_server.py imports this router during its own init).
  from benchmarks.api_server.maxtext_server import _queue_and_wait_for_response

  internal_resp = await _queue_and_wait_for_response(internal_req)

  # _queue_and_wait_for_response returns a ChatCompletionResponse pydantic
  # object (validated by FastAPI's response_model on the OpenAI route),
  # but inside our pipeline it's a plain pydantic instance.
  choice = internal_resp.choices[0]
  text_out = choice.message.content if choice.message else ""
  finish = _to_anthropic_stop_reason(choice.finish_reason)

  return AnthropicMessageResponse(
      id=f"msg_{uuid.uuid4().hex}",
      model=req.model,
      content=[AnthropicTextContent(text=text_out or "")],
      stop_reason=finish,
      usage=AnthropicUsage(
          input_tokens=internal_resp.usage.prompt_tokens,
          output_tokens=internal_resp.usage.completion_tokens,
      ),
  )


# Health: a small unauthenticated ping so callers can verify the
# adapter is wired. Same shape as Anthropic's docs example.
@router.get("/v1/messages/health")
def messages_health() -> dict[str, Any]:
  return {"status": "ok", "adapter": "anthropic_v1_messages", "timestamp": int(time.time())}
