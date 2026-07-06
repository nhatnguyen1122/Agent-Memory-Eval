from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import streamlit as st
from dotenv import load_dotenv


APP_PATH = Path(__file__).resolve()
EVAL_DIR = APP_PATH.parents[1]
WORKSPACE_ROOT = APP_PATH.parents[3]

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from benchmarks.common.llm_client import LLMClient  # noqa: E402
from benchmarks.common.memory_backends import create_memory_backend  # noqa: E402


load_dotenv(EVAL_DIR / ".env", override=False)


CATEGORY_ORDER = ["multi-hop", "single-hop", "open-domain", "temporal"]
DEFAULT_EXPERIMENT_PREFIX = "gptoss-locomo-full-rr-v1"
DEFAULT_LLM_BASE_URL = "https://stream-netmind.viettel.vn/gateway"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b:netmind"
DEFAULT_EMBEDDER_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_EMBEDDER_MODEL = "nvidia/llama-nemotron-embed-1b-v2"


def run_async(coro):
    return asyncio.run(coro)


def env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def has_live_keys() -> bool:
    return bool(get_chat_api_key() and get_embedder_api_key())


def get_chat_api_key() -> str:
    return env_first("LLM_API_KEY", "GPTOSS_API_KEY", "NETMIND_API_KEY")


def get_embedder_api_key() -> str:
    return env_first("MEMORY_EMBEDDER_API_KEY", "NVIDIA_API_KEY")


def get_llm_base_url() -> str:
    return env_first("LLM_BASE_URL", default=DEFAULT_LLM_BASE_URL)


def get_llm_model() -> str:
    return env_first("ANSWERER_MODEL", "MEMORY_MODEL", default=DEFAULT_LLM_MODEL)


def get_embedder_base_url() -> str:
    return env_first("MEMORY_EMBEDDER_BASE_URL", "NVIDIA_BASE_URL", default=DEFAULT_EMBEDDER_BASE_URL)


def get_embedder_model() -> str:
    return env_first("MEMORY_EMBEDDER_MODEL", default=DEFAULT_EMBEDDER_MODEL)


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def answer_tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = answer_tokens(prediction)
    ref_tokens = answer_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, reference: str) -> float:
    pred_tokens = answer_tokens(prediction)
    ref_tokens = answer_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens)
    if precision == 0:
        return 0.0
    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return brevity_penalty * precision


def result_for_cutoff(item: dict[str, Any], cutoff: str) -> dict[str, Any]:
    return (item.get("cutoff_results") or {}).get(cutoff) or {}


def is_correct(item: dict[str, Any], cutoff: str) -> bool:
    judgment = str(result_for_cutoff(item, cutoff).get("judgment", "")).upper()
    return judgment == "CORRECT"


def generated_answer(item: dict[str, Any], cutoff: str) -> str:
    return str(result_for_cutoff(item, cutoff).get("generated_answer", "")).strip()


def short(text: str, max_chars: int = 110) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


@st.cache_data(ttl=30, show_spinner=False)
def load_showcase_examples(
    results_dir: str,
    experiment_prefix: str,
    cutoff: str,
) -> list[dict[str, Any]]:
    results_path = Path(results_dir)
    pattern = re.compile(re.escape(experiment_prefix) + r"-(mem0|a_mem)$")
    by_backend: dict[str, dict[tuple[int, int, str], dict[str, Any]]] = defaultdict(dict)
    project_by_backend: dict[str, str] = {}

    for path in results_path.glob("locomo_results_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        metadata = data.get("metadata", {})
        project_name = str(metadata.get("project_name", ""))
        match = pattern.match(project_name)
        if not match:
            continue
        backend = match.group(1)
        project_by_backend[backend] = project_name
        for item in data.get("evaluations") or []:
            key = (
                int(item.get("conversation_idx", -1)),
                int(item.get("category", -1)),
                str(item.get("question_id", "")),
            )
            by_backend[backend][key] = item

    common = set(by_backend["mem0"]) & set(by_backend["a_mem"])
    examples = []
    for key in common:
        mem0_item = by_backend["mem0"][key]
        amem_item = by_backend["a_mem"][key]
        if not is_correct(mem0_item, cutoff) or is_correct(amem_item, cutoff):
            continue
        reference = str(mem0_item.get("ground_truth_answer", ""))
        mem0_answer = generated_answer(mem0_item, cutoff)
        amem_answer = generated_answer(amem_item, cutoff)
        category = str(mem0_item.get("category_name", "unknown"))
        examples.append(
            {
                "key": key,
                "label": (
                    f"{category} | conv {key[0]} | {key[2]} | "
                    f"{short(mem0_item.get('question', ''), 90)}"
                ),
                "category": category,
                "conversation_idx": key[0],
                "question_id": key[2],
                "question": mem0_item.get("question", ""),
                "ground_truth": reference,
                "evidence": mem0_item.get("evidence", []),
                "reference_date": mem0_item.get("reference_date", ""),
                "user_id": mem0_item.get("user_id", ""),
                "mem0_project": project_by_backend.get("mem0", ""),
                "a_mem_project": project_by_backend.get("a_mem", ""),
                "mem0": mem0_item,
                "a_mem": amem_item,
                "mem0_answer": mem0_answer,
                "a_mem_answer": amem_answer,
                "mem0_f1": 100 * token_f1(mem0_answer, reference),
                "mem0_bleu1": 100 * bleu1(mem0_answer, reference),
                "a_mem_f1": 100 * token_f1(amem_answer, reference),
                "a_mem_bleu1": 100 * bleu1(amem_answer, reference),
            }
        )

    category_rank = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}
    examples.sort(
        key=lambda e: (
            category_rank.get(e["category"], 99),
            e["conversation_idx"],
            e["question_id"],
        )
    )
    return examples


@st.cache_resource(show_spinner=False)
def get_mem0_backend(
    project_name: str,
    storage_root: str,
    memory_model: str,
    memory_base_url: str,
    memory_api_key: str,
    embedder_model: str,
    embedder_base_url: str,
    embedder_api_key: str,
    rpm: int,
):
    args = SimpleNamespace(
        memory_backend="mem0",
        project_name=project_name,
        memory_storage_dir=storage_root,
        memory_model=memory_model,
        memory_base_url=memory_base_url,
        memory_api_key=memory_api_key,
        memory_embedder_model=embedder_model,
        memory_embedder_base_url=embedder_base_url,
        memory_embedder_api_key=embedder_api_key,
        backend="local",
        mem0_host="",
        mem0_api_key="",
        rpm=rpm,
    )
    return create_memory_backend(args, benchmark_name="demo", output_dir="results/demo")


def build_llm_client(model: str, base_url: str, api_key: str, rpm: int) -> LLMClient:
    return LLMClient(
        model=model,
        provider="openai",
        api_key=api_key,
        base_url=base_url,
        rpm=rpm,
        timeout=120,
        max_retries=3,
    )


async def add_memory(backend, user_id: str, text: str) -> dict[str, Any] | None:
    role = "user"
    content = text.strip()
    if ":" in content[:20]:
        maybe_role, maybe_content = content.split(":", 1)
        if maybe_role.strip().lower() in {"user", "assistant", "system"}:
            role = maybe_role.strip().lower()
            content = maybe_content.strip()
    if not content:
        return {"results": []}
    return await backend.add([{"role": role, "content": content}], user_id=user_id)


async def search_memories(backend, user_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
    return await backend.search(query, user_id=user_id, top_k=top_k)


def format_memories(memories: list[dict[str, Any]], limit: int | None = None) -> str:
    selected = memories if limit is None else memories[:limit]
    if not selected:
        return "No relevant memories were retrieved."
    lines = []
    for idx, item in enumerate(selected, start=1):
        score = item.get("score", 0.0)
        lines.append(f"[{idx}] score={score:.4f}\n{item.get('memory', '')}")
    return "\n\n".join(lines)


async def answer_with_memories(
    llm: LLMClient,
    question: str,
    memories: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    history = history or []
    recent_history = "\n".join(
        f"{message['role']}: {message['content']}" for message in history[-8:]
    )
    user_prompt = f"""Retrieved memories:
{format_memories(memories, limit=20)}

Recent chat:
{recent_history or "No prior chat in this session."}

Question:
{question}

Answer using the retrieved memories when they are relevant. If the memories do not contain enough information, say what is missing instead of guessing."""
    return await llm.generate(
        system=(
            "You are a memory-grounded assistant. Answer concisely and cite the "
            "important facts from retrieved memories in plain language."
        ),
        user=user_prompt,
        temperature=0,
        max_tokens=512,
    )


def render_memory_cards(memories: list[dict[str, Any]], limit: int = 5) -> None:
    if not memories:
        st.info("No memories available.")
        return
    for idx, memory in enumerate(memories[:limit], start=1):
        score = memory.get("score", 0.0)
        with st.expander(f"Memory {idx} | score {score:.4f}", expanded=idx <= 2):
            st.write(memory.get("memory", ""))
            metadata = memory.get("metadata") or {}
            if metadata:
                st.caption(f"metadata: {metadata}")


def render_showcase() -> None:
    st.subheader("Cherry-picked benchmark replay")
    st.caption(
        "This panel uses saved LoCoMo result JSONs. It does not call the API unless "
        "you press the live Mem0 answer button."
    )

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        results_dir = st.text_input(
            "Results directory",
            value=str(EVAL_DIR / "results" / "locomo"),
        )
    with col_b:
        experiment_prefix = st.text_input("Experiment prefix", value=DEFAULT_EXPERIMENT_PREFIX)
    with col_c:
        cutoff = st.selectbox("Cutoff", ["top_10", "top_20", "top_50"], index=2)

    if st.button("Refresh saved examples"):
        load_showcase_examples.clear()

    examples = load_showcase_examples(results_dir, experiment_prefix, cutoff)
    if not examples:
        st.warning("No Mem0-correct / A-Mem-wrong examples were found for this cutoff.")
        return

    counts = Counter(example["category"] for example in examples)
    st.success(
        f"Found {len(examples)} examples where Mem0 is CORRECT and A-Mem is not at {cutoff}."
    )
    st.caption(
        "By category: "
        + ", ".join(f"{category}: {counts.get(category, 0)}" for category in CATEGORY_ORDER)
    )

    label_to_example = {example["label"]: example for example in examples}
    selected_label = st.selectbox("Showcase question", list(label_to_example))
    example = label_to_example[selected_label]

    st.markdown("#### Question")
    st.write(example["question"])
    st.markdown("#### Ground truth")
    st.success(example["ground_truth"])
    st.caption(
        f"category={example['category']} | conversation={example['conversation_idx']} | "
        f"question_id={example['question_id']} | evidence={example['evidence']}"
    )

    mem0_result = result_for_cutoff(example["mem0"], cutoff)
    amem_result = result_for_cutoff(example["a_mem"], cutoff)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Mem0")
        st.success("Judge: CORRECT")
        st.metric("F1", f"{example['mem0_f1']:.2f}")
        st.metric("BLEU-1", f"{example['mem0_bleu1']:.2f}")
        st.write(example["mem0_answer"])
        if mem0_result.get("reason"):
            st.caption(f"Judge reason: {mem0_result['reason']}")
    with right:
        st.markdown("#### A-Mem")
        st.error(f"Judge: {amem_result.get('judgment', 'UNKNOWN')}")
        st.metric("F1", f"{example['a_mem_f1']:.2f}")
        st.metric("BLEU-1", f"{example['a_mem_bleu1']:.2f}")
        st.write(example["a_mem_answer"] or "(empty answer)")
        if amem_result.get("reason"):
            st.caption(f"Judge reason: {amem_result['reason']}")

    with st.expander("Retrieved memories used in saved benchmark output", expanded=False):
        m_col, a_col = st.columns(2)
        with m_col:
            st.markdown("##### Mem0 retrieval")
            render_memory_cards((example["mem0"].get("retrieval") or {}).get("search_results", []))
        with a_col:
            st.markdown("##### A-Mem retrieval")
            render_memory_cards((example["a_mem"].get("retrieval") or {}).get("search_results", []))

    st.markdown("#### Optional live replay from saved Mem0 memory state")
    st.caption(
        "This searches the persisted Mem0 benchmark memory for the selected "
        "conversation and generates a fresh answer. It requires chat and embedding API keys."
    )
    if not has_live_keys():
        st.info("Set chat and embedding API keys in the environment to enable live replay.")
        return

    if st.button("Ask selected question live with Mem0"):
        with st.spinner("Searching saved Mem0 memory and generating answer..."):
            backend = get_mem0_backend(
                project_name=example["mem0_project"],
                storage_root=str(EVAL_DIR / ".benchmark_state"),
                memory_model=get_llm_model(),
                memory_base_url=get_llm_base_url(),
                memory_api_key=get_chat_api_key(),
                embedder_model=get_embedder_model(),
                embedder_base_url=get_embedder_base_url(),
                embedder_api_key=get_embedder_api_key(),
                rpm=20,
            )
            top_k = int(cutoff.split("_")[-1])
            memories = run_async(search_memories(backend, example["user_id"], example["question"], top_k))
            llm = build_llm_client(get_llm_model(), get_llm_base_url(), get_chat_api_key(), rpm=20)
            answer = run_async(answer_with_memories(llm, example["question"], memories))
        st.markdown("##### Fresh live answer")
        st.write(answer)
        st.markdown("##### Fresh retrieved memories")
        render_memory_cards(memories, limit=5)


def render_live_chat() -> None:
    st.subheader("Live Mem0 chat")
    st.caption(
        "This creates a small Mem0 memory store for an interactive chat. "
        "It requires chat and embedding API keys in your shell environment."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []

    with st.sidebar:
        st.header("Live settings")
        project_name = st.text_input("Demo project", value="streamlit-mem0-demo")
        user_id = st.text_input("User ID", value="demo_user")
        storage_root = st.text_input("Storage root", value=str(EVAL_DIR / ".demo_state"))
        top_k = st.slider("Retrieved memories", min_value=3, max_value=50, value=10, step=1)
        rpm = st.slider("RPM limit", min_value=1, max_value=60, value=20, step=1)

        st.divider()
        st.caption(f"Chat model: `{get_llm_model()}`")
        st.caption(f"Chat base URL: `{get_llm_base_url()}`")
        st.caption(f"Embedder: `{get_embedder_model()}`")
        st.caption(f"Embedder base URL: `{get_embedder_base_url()}`")
        st.caption(f"Chat key set: `{bool(get_chat_api_key())}`")
        st.caption(f"Embedding key set: `{bool(get_embedder_api_key())}`")

    if not has_live_keys():
        st.warning(
            "Live chat is disabled because the environment is missing a chat key or embedding key. "
            "The benchmark replay tab still works without keys."
        )
        return

    backend = get_mem0_backend(
        project_name=project_name,
        storage_root=storage_root,
        memory_model=get_llm_model(),
        memory_base_url=get_llm_base_url(),
        memory_api_key=get_chat_api_key(),
        embedder_model=get_embedder_model(),
        embedder_base_url=get_embedder_base_url(),
        embedder_api_key=get_embedder_api_key(),
        rpm=rpm,
    )
    llm = build_llm_client(get_llm_model(), get_llm_base_url(), get_chat_api_key(), rpm=rpm)

    seed_text = st.text_area(
        "Add seed memories",
        value=(
            "user: My name is Nhat and I am evaluating memory systems.\n"
            "user: I care about LoCoMo and LongMemEval benchmark results.\n"
            "assistant: You prefer direct, practical answers with exact commands."
        ),
        height=110,
    )
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("Add seed memories"):
            lines = [line.strip() for line in seed_text.splitlines() if line.strip()]
            with st.spinner(f"Adding {len(lines)} memory lines..."):
                for line in lines:
                    run_async(add_memory(backend, user_id, line))
            st.success(f"Added {len(lines)} memory lines.")
    with col_b:
        if st.button("Reset this user"):
            run_async(backend.delete_user(user_id))
            st.session_state.chat_messages = []
            st.session_state.last_retrieved = []
            st.success("Deleted memories for this demo user.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask something that should use memory...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Searching Mem0 memories..."):
            memories = run_async(search_memories(backend, user_id, prompt, top_k=top_k))
            st.session_state.last_retrieved = memories

        with st.spinner("Generating answer..."):
            answer = run_async(
                answer_with_memories(llm, prompt, memories, st.session_state.chat_messages)
            )

        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        with st.spinner("Writing this turn back to Mem0..."):
            run_async(
                backend.add(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    user_id=user_id,
                )
            )

    with st.expander("Last retrieved memories", expanded=True):
        render_memory_cards(st.session_state.last_retrieved, limit=10)


def main() -> None:
    st.set_page_config(
        page_title="Mem0 Memory Demo",
        page_icon="M",
        layout="wide",
    )
    st.title("Mem0 Memory Demo")
    st.caption("Interactive chat plus a LoCoMo cherry-pick comparison against A-Mem.")

    tab_chat, tab_showcase = st.tabs(["Live Mem0 chat", "LoCoMo comparison showcase"])
    with tab_chat:
        render_live_chat()
    with tab_showcase:
        render_showcase()


if __name__ == "__main__":
    main()
