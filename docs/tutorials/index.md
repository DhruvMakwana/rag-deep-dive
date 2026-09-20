# Tutorials & Code

This section is for hands-on material: runnable notebooks, worked implementations of the techniques covered in the deep-dive pages (HyDE, RRF fusion, a minimal CRAG loop, a from-scratch cross-encoder reranker, etc.), and end-to-end mini-projects.

## Available now

- [x] **[Build Naive RAG From Scratch](naive-rag-from-scratch.md)** — PDF → chunk → embed → FAISS → retrieve → generate, no framework, runnable with Anthropic/OpenAI/Ollama
- [x] **[Build Adaptive RAG From Scratch](advanced-rag-from-scratch.md)** — an LLM router picks the right pre-retrieval fix per query, then reranks — ties pre- and post-retrieval techniques into one pipeline, with an honest finding that reranking can undo an earlier stage's fix
- [x] **[Build a Multi-Document Production RAG Pipeline](multi-doc-production-rag.md)** — three real papers (not one), a chunking-strategy comparison, dense-vs-hybrid retrieval, a real vector database with cross-document metadata filtering, and a full RAGAS evaluation at the end — with a real hybrid-search-made-it-worse finding and the same grounding-failure pattern from RAGAS Deep Dive reproduced on a different corpus

## Planned

- [ ] Implement a prompted Self-RAG loop
- [ ] Implement CRAG with a web-search fallback
