# Anthropic: When Your Own Eval Suite Still Lets Regressions Through

Anthropic runs one of the more mature eval-driven-development processes publicly described for an agentic product — Claude Code, its own agentic coding tool. That makes what happened between March and April 2026 unusually valuable as a case study: despite that process, three separate regressions shipped to production, degraded Claude Code's perceived intelligence and behavior, and in one case went undetected internally for weeks. Anthropic published both a description of its eval methodology ("Demystifying evals for AI agents") and a detailed postmortem of the incident ("April 23 postmortem"), and together they form one of the most concrete, dated, and specific public accounts of eval infrastructure failing in a real, sophisticated organization — not a strawman of a company that "didn't test enough."

## The problem

Claude Code's eval process, as Anthropic describes it, evolved in stages. It started with informal dogfooding and internal feedback, then added narrow, targeted evals — concision, file-edit correctness — that check specific, well-defined behaviors. Over time this grew into broader behavioral evals meant to catch fuzzier failure modes, such as "over-engineering." Anthropic maintains **two separate eval suites**: one for quality benchmarking (how good is the model at the task in absolute terms) and one for regression testing (did a change make things worse than before), each combining code-based checks, model-based grading (an LLM acting as judge), and human grading. Anthropic explicitly stresses reading raw transcripts rather than relying on aggregate scores alone, because that's the only way to actually trust what a grader is measuring.

This is, by most standards, a serious and layered eval setup. The postmortem's value is precisely that it shows this kind of setup is still not sufficient on its own — three distinct regressions slipped through it in the space of about seven weeks.

## What happened, and why each one slipped through

**Issue 1: reasoning effort, March 4 to April 7.** Anthropic changed Claude Code's default reasoning effort from "high" to "medium," specifically to reduce latency. This wasn't a blind change — internal evals correctly flagged the tradeoff, showing "slightly lower intelligence, significantly less latency." The eval suite did its job. The failure was a judgment call layered on top of a correct measurement: the team optimized for the axis (latency) that was easier to justify internally, rather than the axis (perceived intelligence) that mattered more to how users actually valued the product. The change shipped anyway and stayed live for roughly five weeks before being reverted on April 7.

**Issue 2: a thinking-cache bug, March 26 to April 10.** A cache-optimization change introduced a bug that cleared the model's "thinking" state every single turn, instead of once per idle session as intended. In practice this made Claude seem to forget context mid-conversation — a subtle but real degradation in coherence. What makes this case distinct is *why* it wasn't caught: a concurrent, internal-only experiment happened to mask the bug in Anthropic's own internal test builds, so employees dogfooding Claude Code weren't seeing the same behavior that external, production users were experiencing. The bug was live in production, actively affecting real users, for two full weeks before anyone connected the reports to its actual cause.

**Issue 3: a tool-call verbosity cap, April 16 to April 20.** A system-prompt instruction was added that capped text between tool calls to 25 words or fewer. This produced a measured **3% performance drop for both Opus 4.6 and Opus 4.7** — but that regression only became visible once broader ablations were run across models. Weeks of internal testing against the existing, narrower eval set had shown no regression at all. The problem wasn't that nobody tested the change; it's that the existing test surface wasn't wide enough to expose a cost that a broader, more systematic ablation later revealed in days.

Anthropic's own summary of the situation is blunt: "neither our internal usage nor evals initially reproduced the issues identified." That sentence is the entire postmortem in miniature — a mature, two-suite, human-plus-model-graded eval process, staffed by the people who literally build the underlying models, still initially failed to catch all three problems.

## What changed afterward

Anthropic's response was a specific, concrete list of process changes rather than a vague commitment to "test more":

- Run a **broad per-model eval suite for every system-prompt change**, not just changes that look targeted or low-risk.
- Continue **systematic, line-by-line ablations** rather than testing changes in aggregate, since aggregate testing is exactly what let Issue 3 hide.
- Build tooling to make **prompt diffs auditable**, so changes to the system prompt are individually traceable.
- **Gate model-specific changes to the model they were tuned for**, so a change validated against one model can't silently affect another it was never tested against.
- Add **soak periods and gradual rollouts** for any change that could plausibly affect perceived intelligence, rather than shipping straight to full production.
- **Increase internal staff usage of the exact public build**, directly closing the dogfood-vs-production gap that let Issue 2 go unnoticed for two weeks.
- **Expand the context window of their AI-based code-review tool**, so it can automatically catch more of this class of issue before it ships.

## The numbers

The timeline is precisely dated: Issue 1 ran March 4 to April 7; Issue 2 ran March 26 to April 10 (overlapping Issue 1); Issue 3 ran April 16 to April 20; the postmortem itself was published April 23, 2026. Issue 3's regression was quantified at a **3% performance drop**, measured identically for both Opus 4.6 and Opus 4.7, and that number only surfaced once ablations were broadened beyond the existing eval set.

!!! success "The lesson"
    This is arguably the best publicly available case study of continuous evaluation not being a solved problem — even inside the company that builds the model being evaluated. A narrow eval suite, or one optimized for the wrong metric (latency over quality, in Issue 1), can let real regressions reach production even when the team reading the results is highly sophisticated. The countermeasures Anthropic adopted — broad per-change ablations instead of aggregate testing, soak periods and gradual rollouts for anything touching perceived quality, gating model-specific changes to that model, and closing the gap between internal dogfooding and the actual public build — are concrete, transferable design patterns for eval infrastructure, not abstract advice to "test more."

## Sources

- [Demystifying evals for AI agents (Anthropic engineering)](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [April 23 postmortem (Anthropic engineering)](https://www.anthropic.com/engineering/april-23-postmortem)
