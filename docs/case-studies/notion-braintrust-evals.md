# Notion: A Dual-Track Evaluation Framework for 70 Engineers

Notion AI has grown from simple prompt-plus-judge setups into genuinely complex agentic systems, and that growth created an organizational problem as much as a technical one: how do roughly 70 engineers stay aligned on what "good" output looks like, and how does the team adopt new frontier models quickly without quietly breaking behavior that already works? Notion's answer, built on top of Braintrust as its eval infrastructure and described by Sarah Sachs, Notion's AI Modeling Lead, is a useful case study precisely because it treats evaluation as an organizational-scale coordination problem, not just a scoring script.

## The problem

At small scale, a single prompt plus a single LLM-judge score is enough to sanity-check quality. That approach breaks down once a product has dozens of engineers touching AI features and the systems themselves become agentic — multi-step, tool-using, and much harder to reduce to one number. Notion needed two things at once: a way to catch regressions before they reached customers, and a way to move fast on new frontier models as they were released, without either goal blocking the other. Left unaddressed, teams in this position tend to pick one side by default — either they get too cautious to upgrade models for fear of breaking something, or they chase new capability and let regressions slip through unnoticed.

## What they built, and why

Sachs started from the ground up: manually reviewing worst-case customer transcripts inside Braintrust to find real, recurring failure patterns rather than guessing at them abstractly. That manual review became the seed for a **dual-track evaluation system** with two independently run suites, deliberately kept separate rather than merged into one combined score:

- **Regression evals** — these exist purely to catch breakage. The bar here is a hard gate: the team holds regression performance at effectively **100%**, meaning a change that fails this suite doesn't ship, full stop.
- **Frontier evals** — these exist to measure something different: whether a new model actually *unlocks new capability* Notion didn't have before. This suite is allowed, and expected, to show incremental gains over time, because its purpose is capability discovery, not safety-net checking.

Keeping these as two distinct suites rather than one blended score is the central design decision. A single combined metric would obscure which question was actually being answered — "did we break something" and "is this better" require different sensitivity, different pass bars, and different responses when they fail.

Supporting that dual-track structure, Notion built custom evaluation UI — embedded iframes — to render Notion-specific data structures (pages, blocks, databases) inside Braintrust's tooling rather than forcing everything into generic text diffs. They also built searchable trace infrastructure specifically to support "needle in the haystack" debugging: finding the one relevant span inside long, multilingual conversation traces where an issue actually originated. And they built labeled datasets targeting specific customer segments — for example, APAC and multilingual users — rather than relying on one generic, English-centric eval set to represent every customer's experience.

The organizational payoff of this infrastructure shows up in how the team spends its time: roughly **80%** of the AI team's time now goes into evaluating feedback and traces, rather than building net-new features against untested assumptions ("blind"). That's a deliberate inversion of where effort typically goes in a fast-moving AI team, and it's presented as a feature of the system, not overhead.

## The numbers

Notion's eval infrastructure keeps **70 engineers** aligned on shared quality standards through common tooling rather than ad hoc review. New frontier models are evaluated and deployed in **under 24 hours** of a model's public release — a speed that's only possible because the regression suite gives the team confidence that adopting a new model won't need to be re-litigated line by line each time. The regression suite is held to a **100% pass** hard gate, while the frontier suite separately and independently tracks incremental capability gains without being blocked by that gate.

Sachs also frames the team's posture toward its own architecture in a way worth quoting directly: being willing to "start over every six months" as underlying model capabilities shift. A concrete example she gives is rebuilding Notion's whole approach around self-healing tool-use once models became capable enough to actually support it — an architecture that wouldn't have made sense to build before the underlying models could carry it.

!!! success "The lesson"
    Separate "did we break anything" from "is the new model actually better" into two distinct, independently run eval suites, rather than one blended score. Conflating them creates a bad tradeoff either way: it either blocks confident model upgrades (fear of an undetected regression) or lets real regressions through unnoticed while chasing frontier capability gains. Sachs's "start over every six months" framing is the complementary half of this lesson — evaluation and architecture assumptions should be revisited on a real, recurring cadence as underlying model capabilities change, not treated as a fixed decision made once and never revisited.

## Sources

- [How Notion evaluates AI at scale across 70 engineers (Braintrust customer story)](https://www.braintrust.dev/customers/notion)
