"""
Documentation-only companion to ragas_deep_dive.py.

Every function below is self-contained (own imports, no cross-function
dependencies) so each one can be extracted and pasted standalone into the
blog. Not run as a script, not kept in sync with ragas_deep_dive.py
automatically.
"""


# --8<-- [start:naive_rag]
def run_naive_rag(question: str, chunks: list, reference: str) -> dict:
    """A real naive RAG call: retrieve top-3 chunks, generate an answer,
    package it into the shape RAGAS needs -- question, answer, the
    retrieved contexts (not just the final answer), and a reference
    (ground-truth) answer for the metrics that need one."""
    import anthropic
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = model.encode(question, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = sims.topk(min(3, len(chunks)))
    retrieved_contexts = [chunks[i] for i in top_k.indices.tolist()]

    client = anthropic.Anthropic()
    context_block = "\n---\n".join(retrieved_contexts)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Answer the question using the context below.\n\nContext:\n{context_block}\n\nQuestion: {question}"}],
    )
    answer = response.content[0].text

    return {"question": question, "answer": answer, "retrieved_contexts": retrieved_contexts, "reference": reference}
# --8<-- [end:naive_rag]


# --8<-- [start:ragas_eval]
def run_ragas_eval(samples: list) -> object:
    """Scores real pipeline output with RAGAS's four core metrics.
    `samples` is a list of dicts shaped like `run_naive_rag`'s return
    value.

    Setup requirements for this ragas release: embeddings go through
    `LangchainEmbeddingsWrapper` around a real `langchain-huggingface`
    embeddings object, the same pattern used for the judge LLM via
    `LangchainLLMWrapper` -- ragas's own `HuggingfaceEmbeddings` wrapper
    fails pydantic validation on direct instantiation. The judge LLM
    wrapper also needs `bypass_temperature=True`, since current Claude
    models (Sonnet 5 and later) no longer accept a `temperature`
    parameter, which `LangchainLLMWrapper` otherwise sets on the
    underlying model before every call by default.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-5"), bypass_temperature=True)
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["retrieved_contexts"],
            reference=s["reference"],
        )
        for s in samples
    ])

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    # Anthropic's per-minute rate limits are tighter than ragas's 16-worker
    # default assumes.
    return evaluate(dataset=dataset, metrics=metrics, run_config=RunConfig(max_workers=4, timeout=180))
# --8<-- [end:ragas_eval]


# --8<-- [start:extended_metrics]
async def run_extended_metrics(sample: dict, judge_model: str = "claude-sonnet-5") -> dict:
    """Scores one sample with four metrics from ragas's modern
    `ragas.metrics.collections` API -- AnswerCorrectness (a weighted
    blend of factual correctness and semantic similarity),
    SemanticSimilarity on its own, FactualCorrectness, and
    NoiseSensitivity. `sample` needs question/answer/reference/
    retrieved_contexts keys, matching `run_naive_rag`'s return shape.

    This API builds its LLM via `llm_factory` + the `instructor`
    library rather than LangChain, so `LangchainLLMWrapper`'s
    `bypass_temperature` flag doesn't apply here -- current Claude
    models reject both `temperature` and `top_p`, and this API has no
    equivalent constructor flag, so both are removed directly from the
    constructed LLM's own `model_args` dict. Needs `instructor>=1.17`;
    older versions fail to parse a response when the model returns a
    ThinkingBlock before its text content.
    """
    from anthropic import AsyncAnthropic
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, FactualCorrectness, NoiseSensitivity, SemanticSimilarity

    def make_judge_llm():
        llm = llm_factory(judge_model, provider="anthropic", client=AsyncAnthropic())
        del llm.model_args["temperature"]
        del llm.model_args["top_p"]
        return llm

    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", use_api=False)

    correctness = AnswerCorrectness(llm=make_judge_llm(), embeddings=embeddings)
    similarity = SemanticSimilarity(embeddings=embeddings)
    factual = FactualCorrectness(llm=make_judge_llm())
    noise = NoiseSensitivity(llm=make_judge_llm())

    answer_correctness = await correctness.ascore(user_input=sample["question"], response=sample["answer"], reference=sample["reference"])
    semantic_similarity = await similarity.ascore(reference=sample["reference"], response=sample["answer"])
    factual_correctness = await factual.ascore(response=sample["answer"], reference=sample["reference"])
    noise_sensitivity = await noise.ascore(
        user_input=sample["question"], response=sample["answer"], reference=sample["reference"], retrieved_contexts=sample["retrieved_contexts"],
    )
    return {
        "answer_correctness": answer_correctness.value,
        "semantic_similarity": semantic_similarity.value,
        "factual_correctness": factual_correctness.value,
        "noise_sensitivity": noise_sensitivity.value,
    }
# --8<-- [end:extended_metrics]


# --8<-- [start:testset_generation]
def generate_testset(document_text: str, testset_size: int = 6, judge_model: str = "claude-sonnet-5") -> object:
    """Generates a synthetic eval set (question + reference answer
    pairs) directly from a source document, for when there's no
    hand-written eval set yet. Uses the legacy LangchainLLMWrapper/
    LangchainEmbeddingsWrapper, not the modern collections API, since
    TestsetGenerator's constructor expects those wrapper types
    specifically. Needs the optional `rapidfuzz` package for its
    internal relationship-builder transform."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.documents import Document as LCDocument
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator

    docs = [LCDocument(page_content=document_text)]
    llm = LangchainLLMWrapper(ChatAnthropic(model=judge_model), bypass_temperature=True)
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    return generator.generate_with_langchain_docs(docs, testset_size=testset_size)
# --8<-- [end:testset_generation]
