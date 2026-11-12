"""
Pluggable LLM providers for the generation step.

Pick a provider with the LLM_PROVIDER env var: "anthropic" | "openai" | "ollama".
Anthropic and OpenAI need an API key (read from .env, never logged or printed).
Ollama runs fully locally and needs no key at all — good for testing the
pipeline with zero cost/signup, as long as you have Ollama installed and
`ollama pull llama3.1` (or similar) has been run once.
"""

import os

# --8<-- [start:provider_dispatch]
def generate(prompt: str, provider: str | None = None, model: str | None = None) -> str:
    """Route to the configured LLM provider and return its text response."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()

    if provider == "anthropic":
        return _generate_anthropic(prompt, model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    if provider == "openai":
        return _generate_openai(prompt, model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "ollama":
        return _generate_ollama(prompt, model or os.environ.get("OLLAMA_MODEL", "llama3.1"))

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', or 'ollama'.")
# --8<-- [end:provider_dispatch]


# --8<-- [start:provider_anthropic]
def _generate_anthropic(prompt: str, model: str) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
# --8<-- [end:provider_anthropic]


# --8<-- [start:provider_openai]
def _generate_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
# --8<-- [end:provider_openai]


# --8<-- [start:provider_ollama]
def _generate_ollama(prompt: str, model: str) -> str:
    import requests

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    response = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]
# --8<-- [end:provider_ollama]
