"""Minimal relevance retriever for the bias compatibility layer."""

from __future__ import annotations

from typing import Iterable

from .bias_model import Bias
from .bias_store import BiasStore


class BiasRetriever:
    def __init__(self, store: BiasStore):
        self.store = store

    def get_relevant_biases(
        self,
        context: str,
        *,
        profile_id: str | None = None,
        task_type: str = "",
        platform: str = "cli",
        available_tools: Iterable[str] = (),
        top_k: int = 3,
    ) -> list[Bias]:
        return self.store.get_relevant_biases(
            context,
            profile_id=profile_id,
            task_type=task_type,
            platform=platform,
            available_tools=available_tools,
            top_k=top_k,
        )
