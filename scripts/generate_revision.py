"""
Generates docs/revision.md: a combined revision page aggregating every
content page's TL;DR bullets, every page's Scenario Check quiz, and
every page's flashcard deck into one page.

This is a maintenance script, not a live plugin -- re-run it manually
after adding/editing a page's TL;DR, quiz, or flashcards/<slug>.json
deck, then rebuild the site.

Usage: python3 scripts/generate_revision.py   (run from the blog/ directory)
"""

import json
import re
from pathlib import Path

BLOG_DIR = Path(__file__).parent.parent
DOCS_DIR = BLOG_DIR / "docs"
FLASHCARDS_DIR = BLOG_DIR / "flashcards"

# Mirrors mkdocs.yml's nav order/grouping for content pages that carry a
# TL;DR + Scenario Check. Keep this in sync with mkdocs.yml when the nav
# changes -- there's no automatic link between the two.
SECTIONS = [
    ("Foundations", [
        ("Naive RAG", "naive-rag.md"),
        ("Advanced RAG — Pre-retrieval", "advanced-rag-pre-retrieval.md"),
        ("Advanced RAG — Post-retrieval", "advanced-rag-post-retrieval.md"),
    ]),
    ("Architectures", [
        ("Modular RAG", "modular-rag.md"),
        ("Self-RAG", "self-rag.md"),
        ("Corrective RAG (CRAG)", "corrective-rag.md"),
        ("Agentic RAG", "agentic-rag.md"),
        ("Graph RAG", "graph-rag.md"),
    ]),
    ("Specialized & Alternative Retrieval", [
        ("Vision-RAG", "vision-rag.md"),
        ("Multimodal RAG", "multimodal-rag.md"),
        ("Video-RAG", "video-rag.md"),
        ("CAG (Cache-Augmented Generation)", "cag.md"),
        ("PageIndex", "pageindex.md"),
    ]),
    ("Retrieval Engineering", [
        ("Chunking Strategies", "chunking.md"),
        ("Embedding Model Selection", "embedding-models.md"),
        ("Retrieval Methods", "retrieval-methods.md"),
        ("Vector Databases", "vector-databases.md"),
    ]),
    ("Evaluation & Production", [
        ("Response Quality & Safety", "response-quality-safety.md"),
        ("Evaluation Metrics", "evaluation-metrics.md"),
        ("RAGAS Deep Dive", "ragas.md"),
        ("Production Considerations", "production-considerations.md"),
    ]),
]

TLDR_RE = re.compile(r'\?\?\? abstract "TL;DR — quick revision"\n((?:    .*\n?)+)')
QUIZ_RE = re.compile(r'<script type="application/json">\n(.*?)\n</script>', re.S)


def extract_tldr(text: str) -> list[str]:
    m = TLDR_RE.search(text)
    if not m:
        return []
    bullets = []
    for line in m.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:])
    return bullets


def extract_questions(text: str, source_title: str, source_file: str) -> list[dict]:
    questions = []
    for m in QUIZ_RE.finditer(text):
        data = json.loads(m.group(1))
        for q in data.get("questions", []):
            tagged = dict(q)
            tagged["source"] = source_title
            tagged["sourceUrl"] = source_file
            questions.append(tagged)
    return questions


def load_flashcards(source_title: str, filename: str) -> list[dict]:
    slug = filename[:-3] if filename.endswith(".md") else filename
    deck_path = FLASHCARDS_DIR / f"{slug}.json"
    if not deck_path.exists():
        return []
    cards = json.loads(deck_path.read_text())
    tagged = []
    for c in cards:
        card = dict(c)
        card["source"] = source_title
        tagged.append(card)
    return tagged


def main() -> None:
    tldr_sections = []
    all_questions = []
    all_cards = []

    for section_name, pages in SECTIONS:
        section_entries = []
        for title, filename in pages:
            text = (DOCS_DIR / filename).read_text()
            bullets = extract_tldr(text)
            if bullets:
                section_entries.append((title, filename, bullets))
            all_questions.extend(extract_questions(text, title, filename))
            all_cards.extend(load_flashcards(title, filename))
        tldr_sections.append((section_name, section_entries))

    lines = [
        "# Revision",
        "",
        "Every page's TL;DR in one place, every page's Scenario Check merged into one combined quiz, and a flashcard deck for fast recall drilling — for a quick pass before an interview instead of clicking through pages one at a time.",
        "",
        "!!! note \"Kept in sync manually\"",
        "    This page is generated from the TL;DR and Scenario Check blocks on every content page by `scripts/generate_revision.py`, re-run after editing any page's TL;DR or quiz — it isn't live-generated on every build.",
        "",
        '=== "All TL;DRs"',
        "",
    ]

    for section_name, entries in tldr_sections:
        if not entries:
            continue
        lines.append(f"    ## {section_name}")
        lines.append("")
        for title, filename, bullets in entries:
            page_slug = filename[:-3] if filename.endswith(".md") else filename
            lines.append(f"    ### [{title}]({page_slug}.md)")
            lines.append("")
            for b in bullets:
                lines.append(f"    - {b}")
            lines.append("")

    lines.append('=== "Combined Scenario Check"')
    lines.append("")
    lines.append(f"    {len(all_questions)} questions from every page on this site, one combined pass instead of opening each page separately. Every question shows which page it's from — go re-read that page for anything you get wrong.")
    lines.append("")
    lines.append('    <div class="quiz-widget" data-title="Combined Scenario Check — All Pages">')
    lines.append('    <script type="application/json">')
    lines.append("    " + json.dumps({"questions": all_questions}, indent=2).replace("\n", "\n    "))
    lines.append("    </script>")
    lines.append("    </div>")
    lines.append("")

    lines.append('=== "Flashcards"')
    lines.append("")
    if all_cards:
        lines.append(f"    {len(all_cards)} flashcards from every page with a deck so far — click a card to flip it, shuffle for random order.")
        lines.append("")
        lines.append('    <div class="flashcard-widget" data-title="Flashcards — All Pages">')
        lines.append('    <script type="application/json">')
        lines.append("    " + json.dumps({"cards": all_cards}, indent=2).replace("\n", "\n    "))
        lines.append("    </script>")
        lines.append("    </div>")
    else:
        lines.append("    No flashcard decks yet.")
    lines.append("")

    out_path = DOCS_DIR / "revision.md"
    out_path.write_text("\n".join(lines) + "\n")
    total_bullets = sum(len(b) for _, entries in tldr_sections for _, _, b in entries)
    print(f"Wrote {out_path} -- {total_bullets} TL;DR bullets across {sum(len(e) for _, e in tldr_sections)} pages, {len(all_questions)} combined quiz questions, {len(all_cards)} flashcards.")


if __name__ == "__main__":
    main()
