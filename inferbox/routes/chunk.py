"""Text chunking endpoint - splits text into chunks by various strategies."""
import re
from typing import Literal
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["chunk"])

# Sentence boundary regex (simple but works for most languages)
_SENTENCE_RE = re.compile(r'(?<=[.!?。!?])\s+(?=[A-Z\u00C0-\u017F\u4E00-\u9FFF])')


class ChunkRequest(BaseModel):
    text: str
    strategy: Literal["chars", "tokens", "sentences", "paragraphs"] = "chars"
    chunk_size: int = 1000
    overlap: int = 100
    tokenizer_model: str | None = None  # for tokens strategy


class ChunkResponse(BaseModel):
    chunks: list[str]
    strategy: str
    count: int


def _chunk_chars(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + size - overlap, start + 1)
    return chunks


def _chunk_paragraphs(text: str, size: int, overlap: int) -> list[str]:
    paras = re.split(r'\n\s*\n', text.strip())
    chunks = []
    current = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > size:
                chunks.extend(_chunk_chars(para, size, overlap))
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _chunk_sentences(text: str, size: int, overlap: int) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    chunks = []
    current = ""
    overlap_sentences: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= size:
            current = (current + " " + sent) if current else sent
            overlap_sentences.append(sent)
            # Keep overlap window manageable
            while sum(len(s) + 1 for s in overlap_sentences) > overlap and len(overlap_sentences) > 1:
                overlap_sentences.pop(0)
        else:
            if current:
                chunks.append(current)
            current = " ".join(overlap_sentences) + " " + sent if overlap_sentences else sent
            overlap_sentences = [sent]
    if current:
        chunks.append(current)
    return chunks


def _chunk_tokens(text: str, size: int, overlap: int, model: str | None) -> list[str]:
    """Chunk by token count using a tokenizer."""
    try:
        from transformers import AutoTokenizer
        tok_model = model or "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(tok_model)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + size, len(token_ids))
            chunk_ids = token_ids[start:end]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
            if end >= len(token_ids):
                break
            start = max(start + size - overlap, start + 1)
        return chunks
    except Exception as e:
        raise HTTPException(500, f"Token chunking failed: {e}")


@router.post("/chunk", response_model=ChunkResponse)
async def chunk(req: ChunkRequest):
    if not req.text:
        return ChunkResponse(chunks=[], strategy=req.strategy, count=0)

    if req.strategy == "chars":
        chunks = _chunk_chars(req.text, req.chunk_size, req.overlap)
    elif req.strategy == "paragraphs":
        chunks = _chunk_paragraphs(req.text, req.chunk_size, req.overlap)
    elif req.strategy == "sentences":
        chunks = _chunk_sentences(req.text, req.chunk_size, req.overlap)
    elif req.strategy == "tokens":
        chunks = _chunk_tokens(req.text, req.chunk_size, req.overlap, req.tokenizer_model)
    else:
        raise HTTPException(400, f"Unknown strategy: {req.strategy}")

    return ChunkResponse(chunks=chunks, strategy=req.strategy, count=len(chunks))
