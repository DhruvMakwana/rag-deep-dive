# LinkedIn: A Knowledge-Graph RAG System That Cut Resolution Time in Production

LinkedIn's customer service organization needed a way for support representatives to query a large corpus of historical support tickets and get accurate, well-grounded answers. The team built a retrieval-augmented system backed by a knowledge graph rather than a conventional vector index, published it as a peer-reviewed paper at SIGIR 2024's industry track, and — unusually for a published RAG paper — reported the actual operational impact of running it in production for six months. That combination of peer review and a real business-metric result makes it one of the more citable case studies for arguing that structure-aware retrieval beats flat text chunking on certain corpora.

## The problem

A support ticket isn't a flat document. It has internal structure — a problem description, a diagnosis, a resolution — and it has relationships to other tickets that share a root cause, a fix, or a product area. Standard RAG treats each ticket as chunks of plain text dropped into a vector index, which discards both of those signals: the chunking process fragments the problem/diagnosis/resolution structure within a single ticket, and a vector index has no native way to represent that ticket A and ticket B are related because they share a root cause, even if their surface text barely overlaps. LinkedIn's team found this measurably hurt both retrieval precision and the quality of the generated answers.

## What they built, and why

Instead of chunking tickets into text for a vector store, LinkedIn built a knowledge graph directly from historical issue-tracking data, explicitly designed to preserve two kinds of structure at once: **intra-issue structure** — the fields and relations within a single ticket — and **inter-issue structure** — the links between related tickets. At query time, the system parses the incoming question, then retrieves the relevant **subgraph** from the knowledge graph, rather than a flat top-k list of text chunks. That subgraph — carrying both a ticket's internal fields and its connections to related tickets — is fed into an LLM as structured context for generation.

The reasoning behind this choice is the paper's core empirical claim: the fields and relationships that standard chunking throws away are often exactly the signal that makes a past ticket relevant to a new question — same root cause, same fix, same product area. A flat vector-similarity search over chunked text has no mechanism to represent "these two tickets are related because of their diagnosis field," even when that relationship is precisely why a rep would want to see both. Retrieving a subgraph instead of a flat ranked list lets the generation step see that relational context explicitly, rather than trying to infer it from unrelated bags of text.

## The numbers

The system delivered a **+77.6% improvement in Mean Reciprocal Rank** for retrieval compared to baseline, and a **+0.32 BLEU** improvement in generation quality compared to baseline. Retrieval was evaluated with MRR, Recall@K, and NDCG@K, and generation with BLEU, ROUGE, and METEOR — a standard multi-metric evaluation protocol rather than a single headline number. What sets this paper apart, though, is that it wasn't left at the IR-metric stage: the system was deployed in LinkedIn's actual customer service team for roughly six months at the time of publication, and in that deployment it produced a **28.6% reduction in median per-issue resolution time** — a real operational outcome, not a proxy metric.

!!! success "The lesson"
    When your documents have inherent internal structure and relational connections to each other — tickets, incidents, cases, threads — naive text-chunking RAG throws away precisely the signal that makes retrieval work well. Modeling the corpus as a graph and retrieving subgraphs, a GraphRAG-style approach that LinkedIn was running in production before that term entered common use, can outperform flat vector retrieval where those relationships matter. It's also a rare example of a published RAG paper that reports an actual downstream business metric — resolution time — rather than stopping at retrieval and generation scores that are easy to report but harder to connect to real value.

## Sources

- [LinkedIn: Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering (arXiv)](https://arxiv.org/abs/2404.17723)
- [ACM: Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering (SIGIR 2024)](https://dl.acm.org/doi/10.1145/3626772.3661370)
