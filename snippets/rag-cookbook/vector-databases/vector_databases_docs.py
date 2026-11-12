"""
Documentation-only companion to vector_databases.py.

Every function below is self-contained (own imports, own constants, no
cross-function dependencies) so each one can be extracted and pasted
standalone into the blog. Not run as a script, not kept in sync with
vector_databases.py automatically.
"""


# --8<-- [start:build_corpus]
def build_corpus(pdf_path) -> list:
    """Chunks a PDF, tagging each chunk with the real page number it
    came from -- this page number is the metadata every database demo
    below filters on."""
    from pypdf import PdfReader

    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50

    reader = PdfReader(str(pdf_path))
    chunks = []
    chunk_id = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece, "page": page_num})
            chunk_id += 1
    return chunks
# --8<-- [end:build_corpus]


# --8<-- [start:faiss_demo]
def demo_faiss(chunks: list, embeddings, query: str, embed_query) -> None:
    """FAISS has no native metadata -- IndexIDMap2 + an external
    {id: page} dict is the standard pattern, and filtering happens via
    IDSelectorBatch passed as `sel=` in SearchParameters at query time.
    `embed_query(query)` returns a flat 1D float32 vector; FAISS wants
    a 2D batch of query vectors even for a single query."""
    import faiss
    import numpy as np

    dim = embeddings.shape[1]
    page_by_id = {c["id"]: c["page"] for c in chunks}
    ids = np.array([c["id"] for c in chunks], dtype="int64")

    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(embeddings, ids)

    query_vec = embed_query(query).reshape(1, -1).astype("float32")

    scores, result_ids = index.search(query_vec, 3)
    print("Unfiltered:", [(page_by_id[cid], round(float(s), 3)) for s, cid in zip(scores[0], result_ids[0])])

    # Metadata filter: restrict the search to a page range using
    # IDSelectorBatch, passed at query time via SearchParametersHNSW.
    allowed_ids = np.array([c["id"] for c in chunks if 7 <= c["page"] <= 9], dtype="int64")
    selector = faiss.IDSelectorBatch(allowed_ids)
    params = faiss.SearchParametersHNSW(sel=selector)
    scores, result_ids = index.search(query_vec, 3, params=params)
    print("Filtered to pages 7-9:", [(page_by_id[cid], round(float(s), 3)) for s, cid in zip(scores[0], result_ids[0]) if cid != -1])
# --8<-- [end:faiss_demo]


# --8<-- [start:faiss_ivf_demo]
def demo_faiss_ivf(chunks: list, embeddings, query: str, embed_query, nlist: int = 8) -> None:
    """IVF clusters vectors around `nlist` centroids (learned via
    k-means in `.train()`) and only searches the `nprobe` closest
    clusters at query time -- `nprobe=nlist` is equivalent to
    exhaustive search over every cluster. IVF's benefits only show up
    at real scale (thousands to millions of vectors); at a small corpus
    the clusters are too sparse to demonstrate a real speed/recall
    trade-off, though the code and training step are identical to
    production usage."""
    import faiss
    import numpy as np

    dim = embeddings.shape[1]
    page_by_id = {c["id"]: c["page"] for c in chunks}
    ids = np.array([c["id"] for c in chunks], dtype="int64")

    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(embeddings)  # k-means over the vectors themselves; required before add()
    index = faiss.IndexIDMap2(ivf)
    index.add_with_ids(embeddings, ids)

    query_vec = embed_query(query).reshape(1, -1).astype("float32")

    ivf.nprobe = 1
    scores, result_ids = index.search(query_vec, 3)
    print("nprobe=1:", [(page_by_id[cid], round(float(s), 3)) for s, cid in zip(scores[0], result_ids[0]) if cid != -1])

    ivf.nprobe = nlist  # exhaustive -- searches every cluster
    scores, result_ids = index.search(query_vec, 3)
    print(f"nprobe={nlist} (exhaustive):", [(page_by_id[cid], round(float(s), 3)) for s, cid in zip(scores[0], result_ids[0]) if cid != -1])
# --8<-- [end:faiss_ivf_demo]


# --8<-- [start:pgvector_ivfflat_demo]
def demo_pgvector_ivfflat(chunks: list, embeddings, query: str, embed_query, dsn: str) -> None:
    """pgvector supports IVFFlat as an alternative to HNSW. `lists` is
    sized per the pgvector README's own guidance: rows/1000 for under
    1M rows, sqrt(rows) above that."""
    import psycopg
    from pgvector.psycopg import register_vector

    dim = embeddings.shape[1]
    lists = max(1, len(chunks) // 1000)
    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS attention_paper_ivf")
        conn.execute(f"CREATE TABLE attention_paper_ivf (id BIGINT PRIMARY KEY, page INT, embedding VECTOR({dim}))")

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO attention_paper_ivf (id, page, embedding) VALUES (%s, %s, %s)",
                [(c["id"], c["page"], embeddings[i]) for i, c in enumerate(chunks)],
            )

        conn.execute(f"CREATE INDEX ON attention_paper_ivf USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})")
        conn.execute("SET ivfflat.probes = 1")

        query_vec = embed_query(query)
        rows = conn.execute(
            "SELECT page, embedding <=> %s AS distance FROM attention_paper_ivf ORDER BY embedding <=> %s LIMIT 3",
            (query_vec, query_vec),
        ).fetchall()
        print(f"lists={lists}, probes=1:", [(page, round(d, 3)) for page, d in rows])
# --8<-- [end:pgvector_ivfflat_demo]


# --8<-- [start:chromadb_demo]
def demo_chromadb(chunks: list, embeddings, query: str, embed_query) -> None:
    """ChromaDB's metadata is native -- a `where` filter alongside the
    vector query -- and HNSW tuning lives in a structured
    `configuration={"hnsw": ...}` dict passed at collection-creation
    time. Current field name is `max_neighbors`, not the more common `M`."""
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="attention_paper",
        configuration={"hnsw": {"space": "cosine", "ef_construction": 200, "ef_search": 64, "max_neighbors": 32}},
    )
    collection.add(
        ids=[str(c["id"]) for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"page": c["page"]} for c in chunks],
        documents=[c["text"] for c in chunks],
    )

    query_vec = [embed_query(query).tolist()]

    result = collection.query(query_embeddings=query_vec, n_results=3)
    print("Unfiltered:", [(m["page"], round(d, 3)) for m, d in zip(result["metadatas"][0], result["distances"][0])])

    # Native metadata filter combined with the vector query in one call.
    result = collection.query(query_embeddings=query_vec, n_results=3, where={"page": {"$in": [8, 9]}})
    print("Filtered to pages [8, 9]:", [(m["page"], round(d, 3)) for m, d in zip(result["metadatas"][0], result["distances"][0])])
# --8<-- [end:chromadb_demo]


# --8<-- [start:qdrant_demo]
def demo_qdrant(chunks: list, embeddings, query: str, embed_query) -> None:
    """`.search()` was removed from the qdrant-client entirely (not just
    deprecated) -- `query_points()` is the only method now. Embedded
    mode (`QdrantClient(":memory:")` or `path=`) needs no server."""
    from qdrant_client import QdrantClient, models

    dim = embeddings.shape[1]
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="attention_paper",
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),
        ),
    )
    client.upsert(
        collection_name="attention_paper",
        points=[
            models.PointStruct(id=c["id"], vector=embeddings[i].tolist(), payload={"page": c["page"]})
            for i, c in enumerate(chunks)
        ],
    )

    query_vec = embed_query(query).tolist()

    result = client.query_points(collection_name="attention_paper", query=query_vec, limit=3)
    print("Unfiltered:", [(p.payload["page"], round(p.score, 3)) for p in result.points])

    # Native metadata filter, e.g. excluding appendix pages via a Range
    # condition on the real page-number payload field.
    result = client.query_points(
        collection_name="attention_paper",
        query=query_vec,
        limit=3,
        query_filter=models.Filter(must=[models.FieldCondition(key="page", range=models.Range(lte=10))]),
    )
    print("Filtered to page <= 10:", [(p.payload["page"], round(p.score, 3)) for p in result.points])
# --8<-- [end:qdrant_demo]


# --8<-- [start:milvus_demo]
def demo_milvus(chunks: list, embeddings, query: str, embed_query, db_path: str) -> None:
    """Milvus Lite is a local, embedded, pure-Python MilvusClient --
    despite the ".db" suffix suggesting a single file, it actually
    creates a directory (WAL + storage), so cleanup needs shutil.rmtree,
    not Path.unlink. Filtering uses a string expression syntax rather
    than a structured filter object."""
    import shutil

    from pymilvus import DataType, MilvusClient

    shutil.rmtree(db_path, ignore_errors=True)
    client = MilvusClient(db_path)

    dim = embeddings.shape[1]
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("page", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 32, "efConstruction": 200})

    client.create_collection(collection_name="attention_paper", schema=schema, index_params=index_params)
    client.insert(
        collection_name="attention_paper",
        data=[{"id": c["id"], "page": c["page"], "vector": embeddings[i].tolist()} for i, c in enumerate(chunks)],
    )

    query_vec = embed_query(query).tolist()

    result = client.search(collection_name="attention_paper", data=[query_vec], limit=3, output_fields=["page"], search_params={"params": {"ef": 64}})
    print("Unfiltered:", [(h["entity"]["page"], round(h["distance"], 3)) for h in result[0]])

    # Filter expression syntax -- a real string expression evaluated
    # against the schema field, not a structured filter object.
    result = client.search(
        collection_name="attention_paper", data=[query_vec], limit=3, output_fields=["page"],
        filter="page >= 7 and page <= 9", search_params={"params": {"ef": 64}},
    )
    print("Filtered 'page >= 7 and page <= 9':", [(h["entity"]["page"], round(h["distance"], 3)) for h in result[0]])
# --8<-- [end:milvus_demo]


# --8<-- [start:pgvector_demo]
def demo_pgvector(chunks: list, embeddings, query: str, embed_query, dsn: str) -> None:
    """Vectors and metadata live as ordinary Postgres columns, so
    filtering is just SQL WHERE combined with ORDER BY ... <=> ... LIMIT
    -- no separate filter API to learn. Needs a running Postgres with
    the pgvector extension, e.g.:
    docker run -d -e POSTGRES_PASSWORD=postgres -p 5433:5432 pgvector/pgvector:pg17
    """
    import psycopg
    from pgvector.psycopg import register_vector

    dim = embeddings.shape[1]
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS attention_paper")
        conn.execute(f"CREATE TABLE attention_paper (id BIGINT PRIMARY KEY, page INT, embedding VECTOR({dim}))")

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO attention_paper (id, page, embedding) VALUES (%s, %s, %s)",
                [(c["id"], c["page"], embeddings[i]) for i, c in enumerate(chunks)],
            )

        conn.execute("CREATE INDEX ON attention_paper USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 200)")
        conn.execute("SET hnsw.ef_search = 64")

        query_vec = embed_query(query)

        rows = conn.execute(
            "SELECT page, embedding <=> %s AS distance FROM attention_paper ORDER BY embedding <=> %s LIMIT 3",
            (query_vec, query_vec),
        ).fetchall()
        print("Unfiltered:", [(page, round(d, 3)) for page, d in rows])

        rows = conn.execute(
            "SELECT page, embedding <=> %s AS distance FROM attention_paper WHERE page BETWEEN 8 AND 9 ORDER BY embedding <=> %s LIMIT 3",
            (query_vec, query_vec),
        ).fetchall()
        print("Filtered WHERE page BETWEEN 8 AND 9:", [(page, round(d, 3)) for page, d in rows])
# --8<-- [end:pgvector_demo]


# --8<-- [start:weaviate_demo]
def demo_weaviate(chunks: list, embeddings, query: str, embed_query) -> None:
    """Vectors are supplied directly (self_provided, skipping Weaviate's
    built-in vectorizer modules since embeddings already exist), and
    HNSW config nests inside Configure.Vectors.self_provided(...) as of
    client >= 4.16 -- not a bare top-level kwarg, a common source of
    stale-tutorial bugs. Needs a running server:
    docker run -d -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\
      -e DEFAULT_VECTORIZER_MODULE=none cr.weaviate.io/semitechnologies/weaviate:1.39.5
    """
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.query import Filter

    client = weaviate.connect_to_local()
    try:
        if client.collections.exists("AttentionPaper"):
            client.collections.delete("AttentionPaper")
        collection = client.collections.create(
            name="AttentionPaper",
            properties=[Property(name="page", data_type=DataType.INT)],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(ef_construction=200, max_connections=32),
            ),
        )

        with collection.batch.dynamic() as batch:
            for i, c in enumerate(chunks):
                batch.add_object(properties={"page": c["page"]}, vector=embeddings[i].tolist(), uuid=weaviate.util.generate_uuid5(c["id"]))

        query_vec = embed_query(query).tolist()

        result = collection.query.near_vector(near_vector=query_vec, limit=3, return_metadata=["distance"])
        print("Unfiltered:", [(o.properties["page"], round(o.metadata.distance, 3)) for o in result.objects])

        result = collection.query.near_vector(
            near_vector=query_vec, limit=3, return_metadata=["distance"],
            filters=Filter.by_property("page").less_or_equal(10),
        )
        print("Filtered to page <= 10:", [(o.properties["page"], round(o.metadata.distance, 3)) for o in result.objects])
    finally:
        client.close()
# --8<-- [end:weaviate_demo]
