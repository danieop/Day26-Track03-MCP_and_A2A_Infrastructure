"""Shared LLM factory for all agents."""

import os

from langchain_openai import ChatOpenAI
from openai import OpenAI


SHOPAIKEY_BASE_URL = "https://api.shopaikey.com/v1"
DEFAULT_MODEL = "gpt-5.4-mini"


def get_openai_client() -> OpenAI:
    """Return the raw OpenAI-compatible ShopAIKey client."""
    return OpenAI(
        api_key=os.getenv("SHOPAIKEY_API_KEY"),
        base_url=os.getenv("SHOPAIKEY_BASE_URL", SHOPAIKEY_BASE_URL),
    )


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at ShopAIKey."""
    return ChatOpenAI(
        model=os.getenv("SHOPAIKEY_MODEL", DEFAULT_MODEL),
        openai_api_key=os.getenv("SHOPAIKEY_API_KEY"),
        openai_api_base=os.getenv("SHOPAIKEY_BASE_URL", SHOPAIKEY_BASE_URL),
    )
