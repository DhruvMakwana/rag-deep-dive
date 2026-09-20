# Response Quality & Safety

RAG systems fail in ways plain chatbots don't, and get evaluated on axes plain chatbots don't need: whether an answer is actually grounded in what was retrieved, whether a document silently smuggled in an instruction the model shouldn't follow, and whether a chunk that belongs to one user leaks into another user's answer. This page covers hallucination mitigation, guardrails, RAG-specific prompt injection, and output/PII safety — current tools and current incidents, not a generic AI-safety checklist.

??? abstract "TL;DR — quick revision"
    - **Aggregate faithfulness scores hide per-claim hallucination** — a response with 5 claims and 1 fabricated detail can still average ~0.8 on a faithfulness metric; the fix is claim-level decomposition (what [RAGAS's Faithfulness metric](ragas.md#the-four-metrics-precise-mechanics) already does), not a single answer-level verdict
    - **Anthropic's Citations API grounds *attribution*, not *truth*** — it guarantees a claim came from an exact source passage, not that the claim is correct; a 2025 OpenAI/Georgia Tech paper argues hallucination persists because training-time benchmarks reward confident guessing over calibrated "I don't know" answers
    - **RAG's prompt-injection surface is the retrieved content itself**, not just the user's message — EchoLeak (CVE-2025-32711) was a real, zero-click production exploit where a hidden instruction in an unopened email got pulled into Microsoft 365 Copilot's context and exfiltrated data
    - **Retrieval-time access control must be a hard metadata filter at the database layer, not a prompt instruction** — the LLM cannot be relied on as a security boundary for content it was never supposed to see in the first place

## Hallucination mitigation in RAG

Giving a model correct retrieved context doesn't guarantee it uses that context faithfully. It can still ignore correct context and answer from parametric memory, blend a real retrieved fact with a fabricated detail in the same sentence, or misattribute a claim to the wrong source. RAG lowers hallucination risk by grounding generation in retrieved evidence — it doesn't eliminate it.

**Aggregate metrics create a false sense of safety.** A response with five claims and one fabricated detail still scores roughly 0.8 on an averaged, answer-level faithfulness check — a dashboard can show 92% aggregate faithfulness while the true per-claim hallucination rate sits around 18%. Current research on this points toward claim-level decomposition rather than a single yes/no verdict per answer, which is exactly how [RAGAS's Faithfulness metric](ragas.md#the-four-metrics-precise-mechanics) works: decompose the response into atomic claims, check each one individually against retrieved context, and score supported-claims ÷ total-claims — a graded signal instead of one binary judgment that loses which specific claim is the problem. A newer variant, `FaithfulnessWithHHEM`, swaps the LLM-as-judge step for [Vectara's HHEM-2.1-Open](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) — a purpose-built entailment classifier — specifically to avoid using an LLM to judge another LLM's output.

**Citation and attribution.** [Anthropic's Citations API](https://platform.claude.com/docs/en/build-with-claude/citations) (GA since mid-2025, also available on AWS Bedrock and other platforms) lets a model return character-level spans pointing to the exact source sentence behind each claim in its answer. The distinction worth being precise about in an interview: this grounds *attribution*, not *truth* — it guarantees "this text came from this exact passage," not "this claim is correct." A model can still misquote or misinterpret a passage it faithfully cites, or cite a passage that only partially supports the specific claim next to it. It's also retrieval-agnostic: the application still has to do retrieval itself; Citations only handles the "show your work" step afterward.

**Uncertainty-based detection.** The original semantic entropy method detects hallucination by sampling many generations for the same query, clustering them by meaning, and measuring entropy across the clusters — high semantic entropy means the model is confabulating rather than converging on an answer. That's expensive (5–10× the inference cost for the extra samples). [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) (Kossen et al., 2024) train a cheap linear probe directly on a single generation's hidden states to approximate the same signal without the sampling overhead. A 2025 RAG-specific extension, [SEReDeEP](https://arxiv.org/pdf/2505.07528), fuses this kind of entropy signal with a second signal distinguishing context-usage from parametric-memory-usage — closer to directly detecting the "ignored good context and answered from memory instead" failure mode this site has measured directly in [RAGAS Deep Dive](ragas.md#putting-the-four-metrics-together-diagnosing-a-rag-system).

!!! note "The root-cause argument: it's a training-incentive problem, not just a missing detector"
    A September 2025 paper from OpenAI and Georgia Tech, ["Why Language Models Hallucinate"](https://arxiv.org/abs/2509.04664), argues hallucinations are ordinary binary classification errors: if a model can't statistically distinguish a true statement from a false one during pretraining, producing false statements is the expected consequence, not a mysterious failure. Its sharper claim is that hallucination *persists* after training because the benchmarks models are optimized against reward confident guessing over admitting uncertainty — like a multiple-choice exam where guessing beats leaving an answer blank. The paper's proposed fix is to change how existing benchmarks are scored (penalize confident wrong answers more than calibrated "I don't know," rather than inventing another hallucination leaderboard) — a training-time, incentive-design fix, in contrast to every other technique on this page, which operates at inference time.

## Guardrails: current tools and mechanics

Guardrails work best as checkpoints at each pipeline stage, not one blanket filter: **input** (validate the query before retrieval — off-topic detection, PII, injection attempts), **retrieval** (access-control filtering as a hard metadata constraint, covered in detail below), **output** (toxicity, PII leakage, policy violations before a response reaches the user), and **tool/action** (for agentic RAG — human-in-the-loop confirmation for anything irreversible).

**NVIDIA NeMo Guardrails** ([docs](https://docs.nvidia.com/nemo/guardrails/), [GitHub](https://github.com/NVIDIA-NeMo/Guardrails)) defines programmable "rails" — input, dialogue, retrieval, output, and execution — using Colang, a small domain-specific language, so a policy like "never discuss competitor pricing" or "reject retrieved content matching an injection pattern" is expressed as a flow the framework checks around every LLM call, rather than baked into the prompt itself.

**Meta's Llama Guard 4** ([model card](https://huggingface.co/meta-llama/Llama-Guard-4-12B)) is the current generation — a 12B multimodal safety classifier that scores both prompts (before the main LLM sees them) and generated responses (after) against a 14-category harm taxonomy, with per-category opt-out. It unified what used to be two separate text- and vision-specific Llama Guard 3 models. **Prompt Guard 2** ([model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)), a much smaller 86M (or 22M, for lower latency) classifier, is specifically for detecting jailbreak/injection-style text — small enough to run as a pre-filter over every retrieved chunk, not just the user's own message, which is exactly the RAG-relevant use case. **Code Shield**, the third piece of Meta's PurpleLlama toolkit, screens LLM-generated code for insecure patterns before it's executed or shown, relevant wherever a RAG system's output feeds into code execution.

**Guardrails AI** ([docs](https://guardrailsai.com/docs)) wraps LLM calls in a `Guard` object with attached `Validator`s (PII detection, toxicity, jailbreak detection, and others) that can `fix`, `reask`, `filter`, or hard-fail on a violation. It's mid-migration away from its hosted Hub toward plain pip-installable validator packages, with the old hosted-inference path fully sunsetting August 25, 2026 — worth checking which install path a given tutorial assumes before following it.

!!! note "Spotlighting: tagging retrieved content as lower-trust, not filtering it out"
    Microsoft's Azure AI Content Safety ships **Prompt Shields**, which separately detects direct attacks (a user trying to override system instructions) and *document* attacks — hidden instructions embedded in third-party content like retrieved documents, emails, or tool output, scanned at both the input and tool-response stage. Its **Spotlighting** feature (preview) is the piece most specific to RAG: it transforms retrieved document content into base64 before it reaches the model, which marks that content as structurally distinct from direct instructions and makes embedded commands harder to parse as authoritative. The trade-offs are real: base64 inflates token count (cost, and can blow context limits on long documents), it's Chat-Completions-API-only, off by default, and the model occasionally starts narrating that content is base64-encoded in its own answers. [Docs →](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)

**OpenAI's Moderation API** is now solely `omni-moderation-latest`, a GPT-4o-based multimodal (text + image) classifier, free to call — the older text-only moderation model was retired in late 2025.

A guardrail layered around a model, rather than baked into its weights, can be independently measured and cost-optimized the same way any other production system can — [Anthropic's own Constitutional Classifiers went through two full generations in production](case-studies/constitutional-classifiers.md), cutting compute overhead roughly 40× while cutting false-positive refusals 87%, with real red-teaming numbers behind both claims.

## Prompt injection: the RAG-specific angle

In a plain chatbot, the only attacker-controlled channel into the model's context is the user's own message — an attacker has no incentive to inject something malicious into a prompt only they will read. RAG changes this: part of the model's context comes from documents the current user never wrote and may not have chosen — search results, scraped pages, other users' shared documents, tool output. Any of those can carry a natural-language instruction planted by an attacker, and unless the pipeline explicitly marks that content as lower-trust (Spotlighting, provenance tracking, or a structural defense like CaMeL below), the model has no built-in way to tell it apart from a legitimate instruction.

This isn't theoretical. The [InjecAgent benchmark](https://aclanthology.org/2024.findings-acl.624/) (ACL 2024), testing tool-using agents against indirect injection delivered through tool output, found ReAct-prompted GPT-4 vulnerable **24% of the time** — and higher when the injected instruction used simple reinforcement tricks like repetition.

!!! success "EchoLeak: the first documented zero-click indirect injection in a production system"
    [CVE-2025-32711](https://arxiv.org/abs/2509.10540), CVSS 9.3 — privately reported to Microsoft in January 2025, patched server-side in May 2025, and publicly disclosed June 11, 2025. An attacker sends a victim a single, unsolicited email containing a prompt-injection payload phrased as ordinary business prose — the victim never has to open or interact with it. Later, when the victim asks Microsoft 365 Copilot an unrelated question, Copilot's own RAG retrieval pulls the malicious email into context because it looks semantically relevant, and the model executes the embedded instructions. The exploit chained four separate bypasses to achieve zero-click exfiltration — evading Microsoft's cross-prompt-injection classifier, defeating link redaction via reference-style Markdown, using an auto-fetched image tag instead of a clickable link, and routing through Microsoft's own CSP-allowlisted Teams proxy — a pattern researchers termed "LLM scope violation": untrusted data influencing behavior beyond the scope it should be confined to. It's the first documented real-world (not red-team) case of RAG-delivered indirect injection achieving actual data exfiltration. [Full case study →](case-studies/echoleak.md)

**CaMeL** ([Google DeepMind, 2025](https://arxiv.org/abs/2503.18813)) is a structural answer to the same problem, not another prompt-level mitigation. It splits the agent into a **Privileged LLM** that only ever sees the trusted user query and produces a plan as restricted, code-like structure (not free text), and a **Quarantined LLM** with no tool-calling capability at all, whose only job is extracting structured values out of untrusted data when the plan calls for it. A custom interpreter attaches provenance metadata ("capabilities") to every value as it flows through the plan, and enforces policies like "never call `send_email` if the recipient's value ever touched untrusted data" — regardless of what the generated code appears to want to do. On the AgentDojo benchmark, CaMeL solved 77% of tasks with provable security guarantees versus 84% for an undefended baseline — a real but bounded capability cost for a structural, not probabilistic, defense.

**Where RAG risk sits in OWASP's current taxonomy.** The [OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/) gives RAG-relevant risk two dedicated categories that didn't exist in the prior list: **LLM01 Prompt Injection** (explicitly naming indirect injection via retrieved content as a subtype), **LLM02 Sensitive Information Disclosure**, and **LLM08 Vector and Embedding Weaknesses** — covering embedding inversion, cross-tenant retrieval leakage, and corpus poisoning. LLM08 being new is itself a signal: OWASP's own view is that RAG's data layer earned a category of its own.

## Output safety and PII in RAG

The RAG-specific version of a PII leak doesn't require the current user's query to contain any PII at all — the leak originates from the retrieved side. A chunk belonging to a different customer or tenant can end up in context, and unless retrieval is permission-filtered, the model will synthesize it into an answer for the wrong person just as readily as it would any other retrieved fact.

**The structurally correct fix is retrieval-time access control**, enforced as a hard metadata filter in the vector search call itself — not a prompt instruction telling the model to "be careful," which relies on the LLM as a security boundary it was never designed to be. Each chunk is tagged at ingestion with metadata mirroring the source system's real ACLs (allowed users, groups, tenant ID), and that metadata filters the similarity search before results ever reach the model — the same metadata-filtering mechanism covered in depth on the [Vector Databases](vector-databases.md#pinecone-cloud-only-documented-here-but-not-run) page, and only as strong as the isolation guarantees of whichever multi-tenancy pattern (namespace-per-tenant, shared index with filtering, or index-per-tenant) that page describes. In practice this is necessary but not sufficient: filters fail silently when a source system's ACLs are stale or incompletely synced into vector metadata, which is why some architectures add a second layer — entity-level re-verification that strips over-permissioned content even after a chunk clears the coarse document-level filter — and choose between pre-filtering (apply the ACL filter before the similarity search, efficient when authorized content is a small fraction of the corpus) and post-filtering (search first, drop unauthorized hits after, fine when it isn't).

**Output-side scanning is defense-in-depth, not a substitute for the above.** [Microsoft Presidio](https://github.com/microsoft/presidio) (open source) runs NER-plus-regex detection and redaction/masking, and the recommended pattern runs it at three points — on the incoming query, on each retrieved chunk before it enters the prompt, and again on the model's final output — rather than trusting a single checkpoint. Presidio's own documentation is explicit that "there is no guarantee Presidio will find all sensitive information," which is exactly why it sits alongside retrieval-time ACLs rather than replacing them. Commercial alternatives (Nightfall AI, AWS Comprehend PII detection, Google Cloud DLP) offer the same detect-and-redact pattern as a managed API, increasingly integrated at the agent-gateway level rather than called ad hoc.

## Scenario Check

<div class="quiz-widget" data-title="Scenario Check: Response Quality & Safety">
<script type="application/json">
{
  "questions": [
    {
      "scenario": "A team's RAG system reports an aggregate faithfulness score of 0.92 across their eval set, and they conclude hallucination isn't a significant problem for their system.",
      "question": "Based on this page, what's the risk with drawing that conclusion from an aggregate score alone?",
      "options": [
        "The score is a direct, reliable ceiling on hallucination -- since faithfulness is computed by averaging per-claim checks, a 0.92 aggregate already guarantees no more than roughly 8% of individual claims across the eval set are unsupported.",
        "An aggregate, answer-level faithfulness score can hide a much higher per-claim hallucination rate, since a response with several correct claims and one fabricated detail still averages to a high score -- claim-level decomposition (like RAGAS's Faithfulness metric) is needed to see the real picture.",
        "The right fix is to raise the faithfulness threshold the team treats as acceptable (e.g. requiring 0.99 instead of 0.92), since the underlying aggregate metric itself is sound.",
        "A 0.92 score means the system is hallucinating on 92% of responses, which the team should treat as a major problem."
      ],
      "correct": 1,
      "explanations": [
        "This inverts what an average tells you: the page's own example shows the opposite can hold -- a 0.92 aggregate is consistent with per-claim hallucination running much higher (its dashboard example: ~92% aggregate alongside a true ~18% per-claim rate), precisely because an average doesn't bound the worst individual cases.",
        "Correct. This page's own example: a response with 5 claims and 1 fabricated detail can still average around 0.8 on an aggregate check, and real dashboards can show ~92% aggregate faithfulness while true per-claim hallucination sits around 18% -- claim-level decomposition is the fix, not the aggregate number itself.",
        "Raising the bar on the same aggregate number doesn't address the actual problem this page identifies -- the issue isn't that 0.92 is too low a bar, it's that any single answer-level average can hide which specific claims are unsupported; the page's fix is claim-level decomposition, not a stricter cutoff on the same kind of score.",
        "A 0.92 aggregate faithfulness score means roughly 92% of decomposed claims were found supported on average -- it does not mean 92% of responses are hallucinated; that inverts what the metric measures."
      ]
    },
    {
      "scenario": "A team adopts Anthropic's Citations API for their RAG system and believes that because every claim in their generated answers now comes with a citation to an exact source passage, factual errors are no longer possible.",
      "question": "Per this page, what's inaccurate about that belief?",
      "options": [
        "Citations API works by having the model re-verify each claim against a live entailment classifier before returning the response, so any claim that fails verification gets suppressed automatically.",
        "Citations API guarantees attribution -- that a piece of text came from an exact source passage -- not truth; the model can still misquote or misinterpret a passage it faithfully cites, or cite a passage that only partially supports the specific claim next to it.",
        "Citations API requires documents to be retrieved through Anthropic's own hosted search index before it can attribute claims to them, so a pipeline doing its own custom retrieval would need to switch to that index first.",
        "Citations API automatically fact-checks every claim against a live web search before returning a response."
      ],
      "correct": 1,
      "explanations": [
        "Citations API doesn't run any truth-verification or entailment check on claims -- it does something narrower: returning character-level spans showing which passage a claim came from. The correct answer's point is that this is attribution, not a verification step that would suppress unsupported claims.",
        "Correct. This page states this distinction explicitly: Citations grounds attribution (this text came from this exact passage), not truth (this claim is correct) -- a model can still misquote or misinterpret a faithfully-cited passage.",
        "This page notes the opposite: Citations API is retrieval-agnostic -- the application still performs its own retrieval however it chooses; Citations only attaches attribution to whatever was actually used in the response, regardless of where or how those documents were retrieved.",
        "Nothing in this page describes Citations API performing live fact-checking -- it attributes claims to source passages already provided to the model, it doesn't verify those passages against external truth."
      ]
    },
    {
      "scenario": "A team building a multi-tenant RAG system decides to handle document-level access control entirely by instructing the model in the system prompt: 'Only answer using information the current user is authorized to see.'",
      "question": "Based on this page's treatment of RAG-specific PII risk, what's the problem with this approach?",
      "options": [
        "This is a reasonable primary control, since instructing the model directly on authorization rules is how this page recommends handling document-level access control.",
        "Prompt instructions rely on the LLM as a security boundary it wasn't designed to be -- the structurally correct fix is a hard metadata filter enforced at the vector search/database layer, so unauthorized chunks are never retrieved into context in the first place.",
        "Relevance-based retrieval already provides a meaningful degree of access control, since a chunk belonging to a different tenant is unlikely to be semantically similar enough to a given user's query to surface in the top results.",
        "Output-side scanning tools like Presidio are the more robust fix here, since catching unauthorized content in the model's final output is a more direct check than filtering earlier at retrieval time."
      ],
      "correct": 1,
      "explanations": [
        "This page frames prompt-instruction-only access control as insufficient, not as the primary recommended mechanism -- the recommended primary control is a hard metadata filter enforced at the vector-search layer, before content ever reaches the model.",
        "Correct. This page states this directly: access control must happen as a hard metadata filter in the vector search call itself, so the LLM never even sees unauthorized content -- not as an instruction the model is trusted to self-enforce.",
        "This page's whole framing for this section is the opposite: a chunk from a different tenant absolutely can be semantically relevant enough to surface in results -- relevance and authorization are independent properties, which is exactly why relying on retrieval's own relevance ranking as an access-control mechanism is the real risk being described.",
        "This page frames it the other way around: output-side scanning (like Presidio) is explicitly defense-in-depth, not the primary or more robust fix -- Presidio's own documentation is quoted as offering no guarantee of catching everything, which is exactly why retrieval-time metadata filtering has to be the structural, primary control."
      ]
    }
  ]
}
</script>
</div>

## Sources & further reading

- [RAGAS Faithfulness metric](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) — the claim-level decomposition approach referenced above, with real measured numbers in [RAGAS Deep Dive](ragas.md)
- [Anthropic Citations API](https://platform.claude.com/docs/en/build-with-claude/citations)
- Kossen et al., [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) (2024)
- [SEReDeEP: Hallucination Detection in RAG via Semantic Entropy](https://arxiv.org/pdf/2505.07528) (2025)
- Kalai, Nachum, Zhang, Vempala, [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) (OpenAI/Georgia Tech, 2025)
- [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/)
- [Meta Llama Guard 4](https://huggingface.co/meta-llama/Llama-Guard-4-12B) and [Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
- [Guardrails AI documentation](https://guardrailsai.com/docs)
- [Azure AI Content Safety: Prompt Shields and Spotlighting](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)
- [InjecAgent benchmark](https://aclanthology.org/2024.findings-acl.624/) (ACL 2024)
- [EchoLeak / CVE-2025-32711 academic writeup](https://arxiv.org/abs/2509.10540)
- Google DeepMind, [CaMeL: Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813) (2025)
- [OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/)
- [Microsoft Presidio](https://github.com/microsoft/presidio)
- [Pinecone: RAG with Access Control](https://www.pinecone.io/learn/rag-access-control/)
- [Vector Databases](vector-databases.md#pinecone-cloud-only-documented-here-but-not-run) — the multi-tenancy isolation patterns this page's access-control section builds on
