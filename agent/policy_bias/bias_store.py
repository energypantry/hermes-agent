"""Minimal bias store wrapper built on top of the policy-bias SQLite store."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .bias_model import (
    BIAS_ID_TO_CANDIDATE_KEY,
    CANDIDATE_KEY_TO_BIAS_ID,
    SEEDED_BIAS_BY_ID,
    SEEDED_BIASES,
    Bias,
)
from .models import PolicyBias, PolicyBiasConfig, derive_profile_id, now_ts
from .retrieval import retrieve_biases
from .scoring import classify_task_type
from .store import PolicyBiasStore
from .synthesis import get_candidate_descriptor


class BiasStore:
    """Small compatibility layer exposing a minimal bias CRUD API."""

    def __init__(
        self,
        *,
        policy_store: Optional[PolicyBiasStore] = None,
        db_path: Optional[Path] = None,
    ):
        self._owns_store = policy_store is None
        self._policy_store = policy_store or PolicyBiasStore(db_path=db_path)

    @property
    def policy_store(self) -> PolicyBiasStore:
        return self._policy_store

    def close(self) -> None:
        if self._owns_store:
            self._policy_store.close()

    def ensure_seed_biases(self, *, profile_id: Optional[str] = None) -> list[Bias]:
        profile_id = profile_id or derive_profile_id()
        seeded: list[Bias] = []
        for bias in SEEDED_BIASES:
            candidate_key = BIAS_ID_TO_CANDIDATE_KEY[bias.id]
            existing = self._policy_store.find_bias_by_candidate_key(profile_id, candidate_key)
            if existing is not None:
                continue
            self.save_bias(bias, profile_id=profile_id)
            seeded.append(bias)
        return seeded

    def save_bias(self, bias: Bias, *, profile_id: Optional[str] = None) -> str:
        profile_id = profile_id or derive_profile_id()
        candidate_key = BIAS_ID_TO_CANDIDATE_KEY.get(bias.id)
        if not candidate_key:
            raise ValueError(f"Unknown bias id: {bias.id}")

        existing = self._policy_store.find_bias_by_candidate_key(profile_id, candidate_key)
        if existing is not None:
            return existing.id

        descriptor = get_candidate_descriptor(candidate_key)
        if descriptor is None:
            raise ValueError(f"No policy-bias descriptor found for {candidate_key}")

        ts = now_ts()
        stored = PolicyBias(
            id=f"seed_{bias.id}",
            profile_id=profile_id,
            scope=descriptor.scope,
            condition_signature=descriptor.condition_signature,
            preferred_policy=descriptor.preferred_policy,
            anti_policy=descriptor.anti_policy,
            rationale_summary=bias.description,
            confidence=float(bias.confidence),
            support_count=1,
            avg_reward=0.8,
            recency_score=1.0,
            decay_rate=0.0,
            status="active",
            source_moment_ids=[],
            created_at=ts,
            updated_at=ts,
            last_triggered_at=None,
            trigger_count=0,
            rollback_parent_id=None,
            version=1,
            bias_candidate_key=candidate_key,
        )
        return self._policy_store.upsert_bias(stored)

    def get_all_biases(self, *, profile_id: Optional[str] = None) -> list[Bias]:
        profile_id = profile_id or derive_profile_id()
        biases = self._policy_store.list_biases(
            profile_id,
            statuses=["active", "shadow", "disabled", "archived"],
            limit=200,
        )
        return [self._to_bias(bias) for bias in biases]

    def get_relevant_biases(
        self,
        context: str,
        *,
        profile_id: Optional[str] = None,
        task_type: str = "",
        platform: str = "cli",
        available_tools: Iterable[str] = (),
        top_k: int = 3,
    ) -> list[Bias]:
        profile_id = profile_id or derive_profile_id()
        self.ensure_seed_biases(profile_id=profile_id)
        available_tools = list(available_tools or [])
        effective_task_type = task_type or classify_task_type(context, available_tools)
        retrieval = retrieve_biases(
            self._policy_store,
            config=PolicyBiasConfig(retrieval_top_k=max(1, int(top_k))),
            profile_id=profile_id,
            user_message=context,
            task_type=effective_task_type,
            platform=platform or "cli",
            available_tools=available_tools,
            include_shadow=False,
        )
        return [self._to_bias(bias) for bias in retrieval.active_biases[: max(1, int(top_k))]]

    @staticmethod
    def _to_bias(bias: PolicyBias) -> Bias:
        seeded = SEEDED_BIAS_BY_ID.get(CANDIDATE_KEY_TO_BIAS_ID.get(bias.bias_candidate_key or "", ""))
        if seeded is not None:
            return Bias(
                id=seeded.id,
                description=seeded.description,
                scope=seeded.scope,
                confidence=float(bias.confidence),
            )
        return Bias(
            id=bias.bias_candidate_key or bias.id,
            description=bias.preferred_policy,
            scope=bias.scope,
            confidence=float(bias.confidence),
        )
