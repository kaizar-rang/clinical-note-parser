"""
Shared LLM wrapper. Swap providers here without touching extractor code.
"""
import json
from config import LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY


def call_llm(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """
    Call the configured LLM and return the response text.

    Args:
        prompt:    User prompt
        system:    System prompt
        json_mode: If True, instruct the model to return valid JSON only

    Returns:
        Raw response string (parse JSON downstream if needed)
    """
    if LLM_PROVIDER == "openai":
        return _call_openai(prompt, system, json_mode)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt, system, json_mode)
    elif LLM_PROVIDER == "local":
        return _call_local(prompt, system, json_mode)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def parse_json_response(response: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def _call_openai(prompt: str, system: str, json_mode: bool) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"model": LLM_MODEL, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_anthropic(prompt: str, system: str, json_mode: bool) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    if json_mode:
        prompt = prompt + "\n\nRespond with valid JSON only. No markdown, no preamble."
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system or "You are a clinical NLP assistant.",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_local(prompt: str, system: str, json_mode: bool) -> str:
    # TODO: implement local model inference (e.g. vLLM, HuggingFace pipeline)
    raise NotImplementedError("Local model inference not yet implemented.")
