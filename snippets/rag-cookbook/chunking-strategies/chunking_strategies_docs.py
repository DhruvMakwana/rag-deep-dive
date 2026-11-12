"""
Documentation-only companion to chunking_strategies.py.

This file is NOT meant to be run top-to-bottom as one script, and it is
NOT imported by anything else in this repo. Its only job is to be sliced
up (by section markers, one function per technique) and embedded in the
blog page — each function below is fully self-contained: its own imports,
its own constant values inlined, so copy-pasting ANY single function out
of this file into a fresh .py works on its own, without needing to know
what else is defined elsewhere.

That's a different goal from chunking_strategies.py, which is intentionally
DRY (shared imports/constants once at the top, reused across functions) —
the right shape for code you actually run and maintain, but a worse one
for a reader looking at one function in isolation on a blog page. Hence
two files: chunking_strategies.py to run, this one to read.

If you change an algorithm in chunking_strategies.py, mirror the change
here too — these are deliberately NOT kept in sync automatically.
"""


# --8<-- [start:fixed_size]
from langchain_text_splitters import CharacterTextSplitter


def fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split every `chunk_size` characters, with `overlap` characters of
    redundancy between consecutive chunks. `separator=""` disables any
    structure-awareness — this is the deliberately naive baseline."""
    splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
# --8<-- [end:fixed_size]


# --8<-- [start:recursive]
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Try splitting on paragraph breaks first; only fall through to a
    finer separator (sentence, then word, then raw character) for any
    piece that's still too large. The reference implementation of this
    technique — most tutorials describing "recursive chunking" mean
    exactly this class."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
# --8<-- [end:recursive]


# --8<-- [start:structure_aware]
from langchain_text_splitters import MarkdownHeaderTextSplitter


def structure_aware_chunk_markdown(markdown_text: str) -> list[dict]:
    """Split on Markdown headers, keeping the heading path as metadata —
    only works when the input actually has real header structure."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(markdown_text)
    return [
        {"heading_path": " > ".join(doc.metadata.values()), "text": doc.page_content}
        for doc in docs
    ]
# --8<-- [end:structure_aware]


# --8<-- [start:semantic]
from chonkie import SemanticChunker


def semantic_chunk(
    text: str, threshold: float = 0.6, embedding_model: str = "all-MiniLM-L6-v2"
) -> list[str]:
    """Embed sentences, walk through in order, and cut a new chunk
    whenever similarity to the next sentence drops below `threshold` (a
    topic shift). Lower threshold = fewer, larger chunks."""
    chunker = SemanticChunker(embedding_model=embedding_model, threshold=threshold)
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]
# --8<-- [end:semantic]


# --8<-- [start:sentence_window]
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser


def sentence_window_chunk(text: str, window_size: int = 2) -> list[dict]:
    """Index each sentence individually (precise matching unit), but keep
    a window of `window_size` sentences on either side for generation
    context, attached as node metadata."""
    parser = SentenceWindowNodeParser.from_defaults(window_size=window_size)
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    return [{"anchor": n.text, "window_text": n.metadata["window"]} for n in nodes]
# --8<-- [end:sentence_window]


# --8<-- [start:small_to_big]
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes


def small_to_big_chunk(text: str, parent_size: int = 1500, child_size: int = 250) -> list[dict]:
    """Embed the small child chunks (precise matching); retrieval fetches
    the linked parent chunk (full context) for generation instead —
    LlamaIndex's HierarchicalNodeParser + get_leaf_nodes is the reference
    implementation, paired with AutoMergingRetriever in a full pipeline."""
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[parent_size, child_size])
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    leaf_nodes = get_leaf_nodes(nodes)
    return [
        {
            "parent_id": leaf.parent_node.node_id if leaf.parent_node else None,
            "child_text": leaf.text,
        }
        for leaf in leaf_nodes
    ]
# --8<-- [end:small_to_big]


# --8<-- [start:late_chunking]
from chonkie import LateChunker


def late_chunk(
    text: str, chunk_size: int = 300, embedding_model: str = "nomic-ai/modernbert-embed-base"
) -> list:
    """Embed the WHOLE document first with a long-context model, so every
    token attends to every other token — THEN pool token embeddings within
    each chunk boundary. Chunk vectors carry context from the rest of the
    document, unlike every other method here, which embeds each chunk in
    isolation."""
    chunker = LateChunker(embedding_model=embedding_model, chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    return [c.embedding for c in chunks]
# --8<-- [end:late_chunking]


# --8<-- [start:proposition]
import json

from transformers import AutoTokenizer, T5ForConditionalGeneration


def proposition_chunk(content: str, title: str = "", section: str = "") -> list[str]:
    """Decompose a passage into propositions — atomic, self-contained
    factual statements — using the actual model released by the Dense X
    Retrieval authors (Chen et al., 2023), fine-tuned specifically for
    this task. Input format ("Title: ... Section: ... Content: ...") and
    JSON-list output are both fixed by how the model was trained."""
    model_name = "chentong00/propositionizer-wiki-flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = f"Title: {title}. Section: {section}. Content: {content}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(input_ids, max_new_tokens=512)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return [output_text]  # model occasionally returns plain text instead of a JSON list
# --8<-- [end:proposition]


# --8<-- [start:agentic]
import os

from chonkie import SlumberChunker
from chonkie.genie import BaseGenie


class AnthropicGenie(BaseGenie):
    """Chonkie's SlumberChunker calls .generate() internally, iteratively,
    to decide where each chunk should end — this adapter is the only
    piece needed to point that at Anthropic (Chonkie ships OpenAI/Gemini/
    Groq/Azure genies built in, just not Anthropic)."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


def agentic_chunk(text: str, chunk_size: int = 400) -> list[str]:
    """Let an LLM decide chunk boundaries directly, using its own reading
    comprehension rather than a mechanical rule. The chunking algorithm
    itself is Chonkie's SlumberChunker; AnthropicGenie above just wires
    in which LLM answers its internal "where should this chunk end"
    questions."""
    chunker = SlumberChunker(genie=AnthropicGenie(), chunk_size=chunk_size, verbose=False)
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]
# --8<-- [end:agentic]


# --8<-- [start:adaptive]
import re

import numpy as np
from chonkie import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _coherence_score(chunks: list[str], embed_model: SentenceTransformer) -> float:
    """Average pairwise sentence-similarity within each chunk — a proxy
    for "do these chunks actually hold together topically?" Higher is
    better."""
    scores = []
    for chunk in chunks:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
        if len(sentences) < 2:
            continue
        embeddings = embed_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        pairwise = [
            _cosine_sim(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        if pairwise:
            scores.append(sum(pairwise) / len(pairwise))
    return sum(scores) / len(scores) if scores else 0.0


def adaptive_chunk(text: str) -> tuple[str, list[str]]:
    """Run multiple candidate strategies — each one a tested library
    implementation, not a custom one — score each for coherence, and
    return whichever wins for THIS document, rather than hardcoding one
    strategy for every document type in a mixed corpus. The selection
    logic itself is necessarily custom (no single library owns this
    meta-technique); what it selects between is not."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    candidates = {
        "fixed_size": CharacterTextSplitter(
            separator="", chunk_size=512, chunk_overlap=50
        ).split_text(text),
        "recursive": RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50
        ).split_text(text),
        "semantic": [
            c.text
            for c in SemanticChunker(embedding_model="all-MiniLM-L6-v2", threshold=0.6).chunk(text)
        ],
    }
    scored = {name: _coherence_score(chunks, embed_model) for name, chunks in candidates.items()}
    best_name = max(scored, key=scored.get)
    return best_name, candidates[best_name]
# --8<-- [end:adaptive]
