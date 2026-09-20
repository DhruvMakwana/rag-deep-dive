# Klarna: An AI Assistant That Replaced 700 Agents — Then Didn't Quite

Klarna is a buy-now-pay-later fintech serving roughly **150 million consumers** across 23 markets, which means its customer-service load is enormous, multilingual, and constant: refunds, returns, payment disputes, balance questions, and payment-schedule changes arrive around the clock in dozens of languages. In February 2024, Klarna announced an AI customer-service assistant built on OpenAI's models and embedded directly in the Klarna app, and framed it not as a chatbot experiment but as a full production replacement for a large slice of its human support operation. It's a useful case study for two reasons: the launch numbers are genuinely strong and come from a first-party press release, and the story didn't end at launch — about a year later, Klarna publicly walked back part of the framing, which is the part most vendor case studies never get to.

## The problem

Support volume at Klarna's scale doesn't scale linearly with headcount in any economically sane way. Hiring enough agents to cover 23 markets, 35+ languages, and 24/7 coverage — while keeping resolution times low and quality consistent — is the kind of problem that pushes a company toward automation not as a novelty but as a capacity necessity. Klarna's explicit goal was to handle the bulk of routine servicing conversations (refunds, returns, disputes, billing questions) without proportionally growing the support org, while preserving a path to a human agent for anything the assistant couldn't or shouldn't close out alone.

## What they built, and why

The design decision that makes this case study interesting is what Klarna chose *not* to build: a triage bot that routes customers to the right queue, or an FAQ deflection layer that answers simple questions and punts everything else to a human. Instead, Klarna built a single conversational assistant intended to **resolve issues end-to-end** — actually processing a refund or resolving a dispute inside the conversation, not just answering questions about the process. The assistant is live 24/7 and can hand off to a human agent on the customer's request, but the human handoff is framed as an escape valve, not the default path.

Klarna's own description of the assistant's capacity was blunt about the ambition: it framed the system as doing **the equivalent of the work of 700 full-time agents**. That's a claim about throughput and headcount substitution, not about incremental support-tooling — a materially different bet than most enterprise chatbot deployments, which are usually pitched as deflection or cost-per-ticket improvements rather than direct labor substitution.

## The numbers

In its first month live, Klarna reported the assistant handled **2.3 million conversations** — about **two-thirds of all Klarna customer-service chats**. Average resolution time dropped from **11 minutes to under 2 minutes**. Repeat inquiries, a reasonable proxy for first-contact resolution, fell **25%**, meaning customers were less likely to have to come back and re-explain the same issue. Klarna reported customer satisfaction as on par with human agents, and estimated the assistant would deliver a **$40 million USD profit improvement** to the business in 2024. All of this ran across 23 markets and 35+ languages, live around the clock, from the first month of deployment.

Roughly a year later, however, Klarna's CEO publicly softened the human-replacement framing and the company began rehiring human agents, citing quality concerns with a fully-automated support model — reported by CX Dive and other press coverage at the time. Klarna didn't disclose a detailed technical postmortem of what specifically degraded, but the public reversal is itself the important data point: a system with genuinely strong, verifiable launch-month numbers still needed human-in-the-loop rebalancing once it had been running in production for a while, past the period any press release covers.

!!! success "The lesson"
    Resolution time and repeat-contact rate are more honest production KPIs for an agentic system than "accuracy" alone, because they measure whether the agent actually closed the loop on a customer's problem rather than just producing a plausible-sounding answer. But this case study is also a warning against reading a launch-month press release as the end of the story: automation-rate and headcount-equivalent numbers, no matter how well-sourced, capture a system's honeymoon period. The real test of an agentic customer-service system is whether its quality holds up over the following year, not the following month — and Klarna's own walk-back is the clearest evidence that a case study should track "what happened next," not stop at the initial announcement.

## Sources

- [Klarna: AI assistant handles two-thirds of customer service chats in its first month](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)
- CX Dive and other press coverage of Klarna's subsequent rehiring of human customer-service agents, roughly a year after the AI assistant's launch
