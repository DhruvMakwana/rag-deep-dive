"""
Documentation-only companion to pageindex_rag.py.

Every function below is self-contained (own imports, no cross-function
dependencies) so each one can be extracted and pasted standalone into the
blog. Not run as a script, not kept in sync with pageindex_rag.py
automatically.
"""


# --8<-- [start:index_document]
def index_document(pdf_path: str) -> str:
    """Build the document's tree. Structure detection (headers, sections,
    table of contents) happens without an LLM where possible -- a layout
    parser reads it directly from the PDF. An LLM is used for two
    smaller things: verifying that a detected heading really starts on
    the page the layout parser thinks it does, and writing each node's
    summary. Returns a doc_id used for every later query against this
    document -- indexing happens once, not per query."""
    from pageindex import PageIndexClient

    client = PageIndexClient(
        index_model="anthropic/claude-haiku-4-5",
        chat_model="anthropic/claude-sonnet-5",
        storage_path="./.pageindex",
    )
    result = client.submit_document(pdf_path)
    return result["doc_id"]
# --8<-- [end:index_document]


# --8<-- [start:pageindex_search]
def pageindex_search(question: str, doc_id: str) -> tuple:
    """Retrieval and generation happen together in one call: the LLM is
    given the document's tree (section titles and summaries, not full
    text), reasons about which section(s) likely contain the answer,
    pulls the actual text for just those sections, and answers --
    looping back for more sections if the first ones weren't enough.
    `citations=True` returns which pages it actually used, so you can
    verify it navigated to the right place rather than trusting the
    answer blindly."""
    import re

    from pageindex import PageIndexClient

    client = PageIndexClient(
        index_model="anthropic/claude-haiku-4-5",
        chat_model="anthropic/claude-sonnet-5",
        storage_path="./.pageindex",
    )
    answer = client.chat(question, doc_id=doc_id, citations=True)
    pages = sorted({int(m) for m in re.findall(r'<cite doc="[^"]*" page="(\d+)"', answer)})
    clean_answer = re.sub(r'\s*<cite doc="[^"]*" page="\d+"/>', "", answer)
    return clean_answer, pages
# --8<-- [end:pageindex_search]


# --8<-- [start:naive_text_search]
def naive_text_search(pdf_path: str, query: str, k: int = 3) -> list:
    """The baseline this page compares against: extract text, chunk it,
    embed chunks with a dedicated text embedder, retrieve by cosine
    similarity. No document structure is used at all -- a chunk from the
    middle of one section looks the same to this pipeline as a chunk
    from the start of another."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer, util

    reader = PdfReader(pdf_path)
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = model.encode(query, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = sims.topk(min(k, len(chunks)))
    return [(chunks[i], sims[i].item()) for i in top_k.indices.tolist()]
# --8<-- [end:naive_text_search]
