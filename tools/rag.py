import threading

import litellm
import pgserver
import psycopg
import yaml
from pgvector import Vector
from pgvector.psycopg import register_vector
import re
from tools.base import ToolEntry

PGDATA_DIR = "./pgdata"
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-001"
DIM = 3072  # must match embedding_model; changing models requires re-ingesting into a fresh ./pgdata

_conn = None
_embedding_config = None
_lock = threading.Lock()

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _hard_split(s: str, target: int) -> list[str]:
    return [s[i : i + target] for i in range(0, len(s), target)]


def chunk_text(text: str, target_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """Split text into chunks of ~target_chars, breaking along paragraph,
    then sentence, then hard character boundaries. Consecutive chunks share
    the previous chunk's last overlap_chars. No chunk exceeds
    target_chars + overlap_chars + 1.
    """
    pieces = []
    for para in _PARAGRAPH_SPLIT.split(text):
        if not para.strip():
            continue
        if len(para) <= target_chars:
            pieces.append(para)
            continue
        for sentence in _SENTENCE_SPLIT.split(para):
            if len(sentence) <= target_chars:
                pieces.append(sentence)
            else:
                pieces.extend(_hard_split(sentence, target_chars))

    chunks = []
    chunk = ""
    for piece in pieces:
        if chunk and len(chunk) + len(piece) > target_chars:
            chunks.append(chunk)
            chunk = chunk[-overlap_chars:]
        chunk += ("\n" + piece) if chunk else piece
    if chunk:
        chunks.append(chunk)
    return chunks

def _get_embedding_config() -> dict:
    global _embedding_config
    if _embedding_config is None:
        with open("config.yaml", encoding="utf8") as f:
            config = yaml.safe_load(f)
        _embedding_config = {
            "model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            "api_base": config.get("embedding_api_base"),
            "api_key": config.get("embedding_api_key"),
        }
    return _embedding_config


def _get_conn() -> psycopg.Connection:
    global _conn
    if _conn is None:
        db = pgserver.get_server(PGDATA_DIR)
        conn = psycopg.connect(db.get_uri(), autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "id bigserial PRIMARY KEY, "
            "content text NOT NULL, "
            f"embedding vector({DIM}))"
        )
        _conn = conn
    return _conn


def get_embedding(text: str) -> Vector:
    response = litellm.embedding(input=[text], **_get_embedding_config())
    return Vector(response.data[0]["embedding"])


def rag_ingest(content: str) -> str:
    embedding = get_embedding(content)
    with _lock:
        _get_conn().execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding),
        )
    return f"Stored document ({len(content)} chars)"


def rag_search(query: str, limit: int = 5) -> str:
    embedding = get_embedding(query)
    with _lock:
        rows = _get_conn().execute(
            "SELECT content, embedding <=> %s AS distance "
            "FROM documents ORDER BY distance LIMIT %s",
            (embedding, limit),
        ).fetchall()
    if not rows:
        return "No documents stored yet"
    return "\n---\n".join(f"[distance {distance:.3f}] {content}" for content, distance in rows)


TOOLS: dict[str, ToolEntry] = {
    "rag_ingest": {
        "function": rag_ingest,
        "schema": {
            "type": "function",
            "function": {
                "name": "rag_ingest",
                "description": "Store a piece of text in long-term memory so it can be retrieved later with rag_search",
                "parameters": {
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Self-contained piece of text to remember",
                        }
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "rag_search": {
        "function": rag_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "Semantically search long-term memory, returning the most similar stored documents (lower distance means more similar)",
                "parameters": {
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return, defaults to 5",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
    },
}
