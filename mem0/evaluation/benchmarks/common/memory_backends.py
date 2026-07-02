"""
Memory backend adapters for benchmark parity.

These adapters let the existing benchmark harness run against multiple memory
systems through the same async contract:

    add(messages, user_id, ...)
    search(query, user_id, top_k=..., ...)
    delete_user(user_id)

The benchmark scripts keep the same ingest/search/evaluate flow and only swap
the backend implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.mem0_client import Mem0Client

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MEM0_REPO = WORKSPACE_ROOT / "mem0"
AMEM_REPO = WORKSPACE_ROOT / "A-mem"

for repo_path in (MEM0_REPO, AMEM_REPO):
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.append(repo_str)


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n".join(parts).strip()


def _epoch_to_ymdhm(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d%H%M")


def _epoch_to_date(timestamp: int | None) -> str:
    if timestamp is None:
        return "unknown"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_source_session_ids(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    values: list[Any] = []
    if metadata.get("source_session_id"):
        values.append(metadata["source_session_id"])
    if metadata.get("source_session_ids"):
        raw = metadata["source_session_ids"]
        values.extend(raw if isinstance(raw, list) else [raw])
    return [str(v) for v in values if v is not None and str(v)]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _metadata_for_vector_store(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert benchmark metadata to vector-store-safe scalar values.

    Chroma rejects lists/dicts in metadata. The benchmark sidecar still keeps
    the original metadata for exact retrieval scoring; this sanitized copy is
    only what gets passed into Mem0's storage path.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False)
    return safe


class BaseMemoryBackend:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        return None

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        raise NotImplementedError

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        raise NotImplementedError

    async def delete_user(self, user_id: str) -> bool:
        raise NotImplementedError

    async def get_user_profile(self, user_id: str) -> dict | None:
        return None


class CurrentMem0Backend(BaseMemoryBackend):
    """Direct in-process adapter for the current Mem0 OSS SDK."""

    def __init__(
        self,
        *,
        graph_enabled: bool,
        storage_dir: Path,
        model: str | None,
        embedder_model: str | None,
        api_key: str | None,
        base_url: str | None,
    ):
        os.environ.setdefault("MEM0_DIR", str(storage_dir / ".mem0"))
        from mem0.memory.main import Memory
        import mem0.memory.main as mem0_memory_main

        self.graph_enabled = graph_enabled
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._patched_symbols: dict[str, Any] = {}
        if not graph_enabled:
            self._patched_symbols = {
                "extract_entities": mem0_memory_main.extract_entities,
                "extract_entities_batch": mem0_memory_main.extract_entities_batch,
                "lemmatize_for_bm25": mem0_memory_main.lemmatize_for_bm25,
            }
            mem0_memory_main.extract_entities = lambda text: []
            mem0_memory_main.extract_entities_batch = lambda texts, batch_size=32: [[] for _ in texts]
            mem0_memory_main.lemmatize_for_bm25 = lambda text: text

        llm_config: dict[str, Any] = {
            "provider": "openai",
            "config": {
                "model": model or os.getenv("MEMORY_MODEL") or "meta/llama-3.1-70b-instruct",
                "temperature": 0.1,
            },
        }
        embedder_config: dict[str, Any] = {
            "provider": "openai",
            "config": {
                "model": embedder_model or os.getenv("MEMORY_EMBEDDER_MODEL") or "nvidia/nv-embedqa-e5-v5",
                "memory_add_embedding_type": os.getenv("MEMORY_ADD_EMBEDDING_TYPE", "passage"),
                "memory_update_embedding_type": os.getenv("MEMORY_UPDATE_EMBEDDING_TYPE", "passage"),
                "memory_search_embedding_type": os.getenv("MEMORY_SEARCH_EMBEDDING_TYPE", "query"),
            },
        }
        if api_key:
            llm_config["config"]["api_key"] = api_key
            embedder_config["config"]["api_key"] = api_key
        if base_url:
            llm_config["config"]["openai_base_url"] = base_url
            embedder_config["config"]["openai_base_url"] = base_url

        config = {
            "version": "v1.1",
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": str(self.storage_dir / "chroma"),
                    "collection_name": "memories",
                },
            },
            "llm": llm_config,
            "embedder": embedder_config,
        }
        self.memory = Memory.from_config(config)
        self._source_metadata_path = self.storage_dir / "source_metadata.json"
        self._source_metadata = self._load_source_metadata()

    def _load_source_metadata(self) -> dict[str, dict[str, Any]]:
        if not self._source_metadata_path.exists():
            return {}
        try:
            data = json.loads(self._source_metadata_path.read_text())
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _persist_source_metadata(self) -> None:
        self._source_metadata_path.write_text(
            json.dumps(self._source_metadata, ensure_ascii=False, indent=2)
        )

    async def close(self) -> None:
        if self._patched_symbols:
            import mem0.memory.main as mem0_memory_main

            for name, original in self._patched_symbols.items():
                setattr(mem0_memory_main, name, original)
            self._patched_symbols = {}

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        add_metadata = dict(metadata or {})
        if observation_date:
            add_metadata["benchmark_observation_date"] = observation_date
        if timestamp is not None:
            add_metadata["benchmark_timestamp"] = timestamp
        vector_store_metadata = _metadata_for_vector_store(add_metadata)

        def _run_add(metadata_for_mem0: dict[str, Any] | None):
            return self.memory.add(
                messages,
                user_id=user_id,
                metadata=metadata_for_mem0,
                prompt=custom_instructions,
            )

        try:
            response = await asyncio.to_thread(_run_add, vector_store_metadata or None)
        except Exception as exc:
            logger.error(
                "CurrentMem0Backend.add failed with metadata for user_id=%s: %s",
                user_id,
                exc,
            )
            logger.debug("%s", traceback.format_exc())
            if vector_store_metadata:
                try:
                    response = await asyncio.to_thread(_run_add, None)
                    logger.warning(
                        "CurrentMem0Backend.add recovered by retrying without vector-store metadata "
                        "for user_id=%s",
                        user_id,
                    )
                except Exception as retry_exc:
                    logger.error(
                        "CurrentMem0Backend.add failed without metadata for user_id=%s: %s",
                        user_id,
                        retry_exc,
                    )
                    logger.debug("%s", traceback.format_exc())
                    return None
            else:
                return None

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        results = []
        metadata_changed = False
        for item in raw_results:
            memory_id = item.get("id", "")
            if memory_id and add_metadata:
                self._source_metadata[memory_id] = dict(add_metadata)
                metadata_changed = True
            results.append(
                {
                    "id": memory_id,
                    "event": item.get("event", "ADD"),
                    "memory": item.get("memory", item.get("data", "")),
                }
            )
        if metadata_changed:
            self._persist_source_metadata()
        return {"results": results}

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        def _run_search():
            return self.memory.search(
                query,
                top_k=top_k,
                filters={"user_id": user_id},
                threshold=0.0,
                rerank=rerank,
                explain=score_debug,
            )

        try:
            response = await asyncio.to_thread(_run_search)
        except Exception as exc:
            logger.error("CurrentMem0Backend.search failed for user_id=%s: %s", user_id, exc)
            logger.debug("%s", traceback.format_exc())
            return []

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        results = []
        for item in raw_results:
            memory_id = item.get("id", "")
            item_metadata = item.get("metadata") or self._source_metadata.get(memory_id, {})
            source_session_ids = _extract_source_session_ids(item_metadata)
            normalized = {
                "id": memory_id,
                "memory": item.get("memory", item.get("data", "")),
                "score": item.get("score", 0.0),
            }
            if item_metadata:
                normalized["metadata"] = item_metadata
            if source_session_ids:
                normalized["source_session_ids"] = source_session_ids
            score_details = item.get("score_details")
            if score_details:
                normalized["score_debug"] = {
                    "combined_score": item.get("score", 0.0),
                    "semantic_score": score_details.get("semantic_score", 0.0),
                    "bm25_score": score_details.get("bm25_score", 0.0),
                    "entity_boost": score_details.get("entity_boost", 0.0),
                }
            results.append(normalized)
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results

    async def delete_user(self, user_id: str) -> bool:
        try:
            await asyncio.to_thread(self.memory.delete_all, user_id=user_id)
            return True
        except Exception:
            return False


class AMemBackend(BaseMemoryBackend):
    """Adapter around AgenticMemorySystem with backend-owned persistence."""

    def __init__(
        self,
        *,
        storage_dir: Path,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
    ):
        from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote

        self.AgenticMemorySystem = AgenticMemorySystem
        self.MemoryNote = MemoryNote
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.getenv("MEMORY_MODEL") or "meta/llama-3.1-70b-instruct"
        self.api_key = api_key
        self.base_url = base_url
        self._systems: dict[str, Any] = {}
        self._source_metadata: dict[str, dict[str, dict[str, Any]]] = {}

    def _state_path(self, user_id: str) -> Path:
        return self.storage_dir / f"{user_id}.json"

    def _build_system(self):
        return self.AgenticMemorySystem(
            llm_backend="openai",
            llm_model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _serialize_system(self, system: Any, user_id: str) -> list[dict[str, Any]]:
        serialized = []
        user_metadata = self._source_metadata.get(user_id, {})
        for note in system.memories.values():
            serialized.append(
                {
                    "id": note.id,
                    "content": note.content,
                    "keywords": note.keywords,
                    "links": note.links,
                    "retrieval_count": note.retrieval_count,
                    "timestamp": note.timestamp,
                    "last_accessed": note.last_accessed,
                    "context": note.context,
                    "evolution_history": note.evolution_history,
                    "category": note.category,
                    "tags": note.tags,
                    "source_metadata": user_metadata.get(note.id, {}),
                }
            )
        return serialized

    def _restore_system(self, user_id: str) -> Any:
        system = self._build_system()
        state_path = self._state_path(user_id)
        if not state_path.exists():
            return system

        data = json.loads(state_path.read_text())
        self._source_metadata.setdefault(user_id, {})
        for item in data:
            source_metadata = item.pop("source_metadata", {}) or {}
            note = self.MemoryNote(**item)
            if source_metadata:
                self._source_metadata[user_id][note.id] = source_metadata
            system.memories[note.id] = note
            metadata = {
                "id": note.id,
                "content": note.content,
                "keywords": note.keywords,
                "links": note.links,
                "retrieval_count": note.retrieval_count,
                "timestamp": note.timestamp,
                "last_accessed": note.last_accessed,
                "context": note.context,
                "evolution_history": note.evolution_history,
                "category": note.category,
                "tags": note.tags,
                "source_metadata": source_metadata,
            }
            system.retriever.add_document(note.content, metadata, note.id)
        return system

    def _get_system(self, user_id: str) -> Any:
        system = self._systems.get(user_id)
        if system is None:
            system = self._restore_system(user_id)
            self._systems[user_id] = system
        return system

    def _persist_system(self, user_id: str, system: Any) -> None:
        self._state_path(user_id).write_text(
            json.dumps(self._serialize_system(system, user_id), ensure_ascii=False, indent=2)
        )

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        system = self._get_system(user_id)
        content = _messages_to_text(messages)
        if not content:
            return {"results": []}

        note_time = _epoch_to_ymdhm(timestamp)
        add_metadata = dict(metadata or {})
        if observation_date:
            add_metadata["benchmark_observation_date"] = observation_date
        if timestamp is not None:
            add_metadata["benchmark_timestamp"] = timestamp

        def _run_add():
            memory_id = system.add_note(content, time=note_time)
            if add_metadata:
                self._source_metadata.setdefault(user_id, {})[memory_id] = add_metadata
            self._persist_system(user_id, system)
            return memory_id

        try:
            memory_id = await asyncio.to_thread(_run_add)
        except Exception:
            return None

        return {"results": [{"id": memory_id, "event": "ADD", "memory": content}]}

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        system = self._get_system(user_id)

        try:
            raw_results = await asyncio.to_thread(system.search_agentic, query, top_k)
        except Exception:
            return []

        formatted = []
        user_metadata = self._source_metadata.get(user_id, {})
        for item in raw_results:
            raw_score = float(item.get("score", 0.0) or 0.0)
            similarity = 1.0 / (1.0 + max(raw_score, 0.0))
            memory_id = item.get("id", "")
            item_metadata = user_metadata.get(memory_id, {})
            source_session_ids = _extract_source_session_ids(item_metadata)
            formatted.append(
                {
                    "id": memory_id,
                    "memory": item.get("content", ""),
                    "score": similarity,
                    "metadata": item_metadata,
                    "source_session_ids": source_session_ids,
                }
            )
        formatted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return formatted[:top_k]

    async def delete_user(self, user_id: str) -> bool:
        self._systems.pop(user_id, None)
        state_path = self._state_path(user_id)
        if state_path.exists():
            state_path.unlink()
        return True


@dataclass
class MemoryBankEntry:
    memory_id: str
    text: str
    date: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryBankBackend(BaseMemoryBackend):
    """
    Lightweight benchmark adapter based on MemoryBank's date-grouped retrieval.

    The original repo is demo- and UI-heavy; for LoCoMo/LongMemEval we only need
    its retrieval shape. This adapter keeps MemoryBank's key benchmark-relevant
    behavior:
      - chunk memories are stored with dates
      - retrieval is dense similarity over stored memory snippets
      - retrieved snippets are grouped by date into memory blocks
    """

    def __init__(self, *, storage_dir: Path, embedding_model: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = (
            embedding_model or os.getenv("MEMORYBANK_EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
        )
        self.encoder = SentenceTransformer(self.embedding_model_name)
        self._entries: dict[str, list[MemoryBankEntry]] = {}

    def _state_path(self, user_id: str) -> Path:
        return self.storage_dir / f"{user_id}.json"

    def _load_entries(self, user_id: str) -> list[MemoryBankEntry]:
        entries = self._entries.get(user_id)
        if entries is not None:
            return entries

        state_path = self._state_path(user_id)
        if not state_path.exists():
            entries = []
        else:
            raw = json.loads(state_path.read_text())
            entries = [MemoryBankEntry(**{**item, "metadata": item.get("metadata", {})}) for item in raw]
        self._entries[user_id] = entries
        return entries

    def _persist_entries(self, user_id: str, entries: list[MemoryBankEntry]) -> None:
        data = [
            {
                "memory_id": entry.memory_id,
                "text": entry.text,
                "date": entry.date,
                "embedding": entry.embedding,
                "metadata": entry.metadata,
            }
            for entry in entries
        ]
        self._state_path(user_id).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        content = _messages_to_text(messages)
        if not content:
            return {"results": []}

        date = observation_date or _epoch_to_date(timestamp)
        entry_id = f"{user_id}_{len(self._load_entries(user_id))}"
        embedding = await asyncio.to_thread(lambda: self.encoder.encode(content).tolist())
        add_metadata = dict(metadata or {})
        if observation_date:
            add_metadata["benchmark_observation_date"] = observation_date
        if timestamp is not None:
            add_metadata["benchmark_timestamp"] = timestamp
        entry = MemoryBankEntry(
            memory_id=entry_id,
            text=content,
            date=date,
            embedding=embedding,
            metadata=add_metadata,
        )
        entries = self._load_entries(user_id)
        entries.append(entry)
        self._persist_entries(user_id, entries)
        return {"results": [{"id": entry_id, "event": "ADD", "memory": content}]}

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        entries = self._load_entries(user_id)
        if not entries:
            return []

        query_embedding = await asyncio.to_thread(lambda: self.encoder.encode(query).tolist())
        scored = []
        for entry in entries:
            score = _cosine_similarity(query_embedding, entry.embedding)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_entries = scored[:top_k]

        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for score, entry in top_entries:
            if entry.date not in grouped:
                source_session_ids = _extract_source_session_ids(entry.metadata)
                grouped[entry.date] = {
                    "id": entry.memory_id,
                    "memory": entry.text,
                    "score": score,
                    "date": entry.date,
                    "metadata": {
                        **entry.metadata,
                        "source_session_ids": source_session_ids,
                        "source_entries": [entry.metadata] if entry.metadata else [],
                    },
                    "source_session_ids": source_session_ids,
                }
                order.append(entry.date)
            else:
                grouped[entry.date]["memory"] += f"\n{entry.text}"
                grouped[entry.date]["score"] = max(grouped[entry.date]["score"], score)
                existing_ids = grouped[entry.date].get("source_session_ids", [])
                merged_ids = _dedupe_preserve_order(
                    existing_ids + _extract_source_session_ids(entry.metadata)
                )
                grouped[entry.date]["source_session_ids"] = merged_ids
                grouped[entry.date]["metadata"]["source_session_ids"] = merged_ids
                if entry.metadata:
                    grouped[entry.date]["metadata"]["source_entries"].append(entry.metadata)

        return [grouped[date] for date in order]

    async def delete_user(self, user_id: str) -> bool:
        self._entries.pop(user_id, None)
        state_path = self._state_path(user_id)
        if state_path.exists():
            state_path.unlink()
        return True


def create_memory_backend(
    args: Any,
    *,
    benchmark_name: str,
    output_dir: str,
) -> BaseMemoryBackend:
    backend_name = getattr(args, "memory_backend", "mem0_server")
    if backend_name == "mem0_server":
        backend = os.getenv("MEM0_BACKEND", args.backend)
        return Mem0Client(
            mode=backend,
            host=args.mem0_host,
            api_key=args.mem0_api_key if backend == "cloud" else None,
            rpm=args.rpm,
        )

    base_storage_dir = Path(getattr(args, "memory_storage_dir", ".benchmark_state"))
    storage_dir = base_storage_dir / benchmark_name / args.project_name / backend_name
    memory_api_key = getattr(args, "memory_api_key", None) or os.getenv("MEMORY_API_KEY")
    memory_base_url = getattr(args, "memory_base_url", None) or os.getenv("MEMORY_BASE_URL")
    memory_model = getattr(args, "memory_model", None) or os.getenv("MEMORY_MODEL")
    memory_embedder_model = getattr(args, "memory_embedder_model", None) or os.getenv("MEMORY_EMBEDDER_MODEL")

    if backend_name == "mem0":
        return CurrentMem0Backend(
            graph_enabled=False,
            storage_dir=storage_dir,
            model=memory_model,
            embedder_model=memory_embedder_model,
            api_key=memory_api_key,
            base_url=memory_base_url,
        )
    if backend_name == "mem0_graph":
        return CurrentMem0Backend(
            graph_enabled=True,
            storage_dir=storage_dir,
            model=memory_model,
            embedder_model=memory_embedder_model,
            api_key=memory_api_key,
            base_url=memory_base_url,
        )
    if backend_name == "a_mem":
        return AMemBackend(
            storage_dir=storage_dir,
            model=memory_model,
            api_key=memory_api_key,
            base_url=memory_base_url,
        )
    if backend_name == "memorybank":
        return MemoryBankBackend(
            storage_dir=storage_dir,
            embedding_model=memory_embedder_model,
        )

    raise ValueError(f"Unsupported memory backend: {backend_name}")


def cleanup_backend_state(storage_root: str | Path) -> None:
    root = Path(storage_root)
    if root.exists():
        shutil.rmtree(root)
