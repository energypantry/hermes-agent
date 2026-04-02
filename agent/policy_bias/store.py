"""SQLite-backed store for policy-bias moments, biases, and traces."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

from hermes_constants import get_hermes_home

from .migrations import BASE_SCHEMA_SQL, BASE_SCHEMA_VERSION, MIGRATIONS, SCHEMA_VERSION
from .models import BiasHistoryEntry, DecisionTrace, PolicyBias, PolicyMoment, new_id

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "policy_bias.db"


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def _json_loads(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


class PolicyBiasStore:
    """SQLite state store for the Policy Bias Engine."""

    _WRITE_MAX_RETRIES = 12
    _WRITE_RETRY_MIN_S = 0.015
    _WRITE_RETRY_MAX_S = 0.120

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    def _execute_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                        return result
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
            except sqlite3.OperationalError as exc:
                err = str(exc).lower()
                if "locked" in err or "busy" in err:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        time.sleep(
                            random.uniform(
                                self._WRITE_RETRY_MIN_S,
                                self._WRITE_RETRY_MAX_S,
                            )
                        )
                        continue
                raise
        raise last_err or sqlite3.OperationalError("database is locked after retries")

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(BASE_SCHEMA_SQL)
        row = cursor.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (BASE_SCHEMA_VERSION,),
            )
            current_version = BASE_SCHEMA_VERSION
        else:
            current_version = int(row["version"])

        while current_version < SCHEMA_VERSION:
            next_version = current_version + 1
            for statement in MIGRATIONS.get(next_version, []):
                try:
                    cursor.execute(statement)
                except sqlite3.OperationalError as exc:
                    # Column-exists and idempotent migration cases fail open.
                    logger.debug(
                        "Policy bias migration v%s statement skipped: %s",
                        next_version,
                        exc,
                    )
            cursor.execute("UPDATE schema_version SET version = ?", (next_version,))
            current_version = next_version

        self._conn.commit()

    def add_moment(self, moment: PolicyMoment) -> str:
        def _do(conn: sqlite3.Connection) -> str:
            conn.execute(
                """
                INSERT OR REPLACE INTO moments (
                    id, profile_id, session_id, timestamp, task_type, platform,
                    context_summary, action_trace_summary, tool_path, decision_class,
                    outcome_class, reward_score, confidence_score, user_feedback_signal,
                    error_signal, side_effect_level, latency_ms, cost_estimate,
                    evidence_refs, extracted_tags, bias_candidate_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    moment.id,
                    moment.profile_id,
                    moment.session_id,
                    moment.timestamp,
                    moment.task_type,
                    moment.platform,
                    moment.context_summary,
                    moment.action_trace_summary,
                    moment.tool_path,
                    moment.decision_class,
                    moment.outcome_class,
                    float(moment.reward_score),
                    float(moment.confidence_score),
                    float(moment.user_feedback_signal),
                    float(moment.error_signal),
                    moment.side_effect_level,
                    moment.latency_ms,
                    moment.cost_estimate,
                    _json_dumps(moment.evidence_refs),
                    _json_dumps(moment.extracted_tags),
                    moment.bias_candidate_key,
                ),
            )
            return moment.id

        return self._execute_write(_do)

    def get_moment(self, moment_id: str) -> Optional[PolicyMoment]:
        row = self._conn.execute(
            "SELECT * FROM moments WHERE id = ?",
            (moment_id,),
        ).fetchone()
        return self._moment_from_row(row) if row else None

    def list_recent_moments(
        self,
        profile_id: str,
        *,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> list[PolicyMoment]:
        sql = "SELECT * FROM moments WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._moment_from_row(row) for row in rows]

    def get_moments_by_candidate(
        self,
        profile_id: str,
        candidate_key: str,
        *,
        limit: int = 100,
    ) -> list[PolicyMoment]:
        rows = self._conn.execute(
            """
            SELECT * FROM moments
            WHERE profile_id = ? AND bias_candidate_key = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (profile_id, candidate_key, max(1, int(limit))),
        ).fetchall()
        return [self._moment_from_row(row) for row in rows]

    def iter_all_moments(self, profile_id: str) -> list[PolicyMoment]:
        rows = self._conn.execute(
            "SELECT * FROM moments WHERE profile_id = ? ORDER BY timestamp ASC",
            (profile_id,),
        ).fetchall()
        return [self._moment_from_row(row) for row in rows]

    def upsert_bias(self, bias: PolicyBias) -> str:
        existing = None
        if bias.id:
            existing = self.get_bias(bias.id)
        if existing is None and bias.bias_candidate_key:
            existing = self.find_bias_by_candidate_key(bias.profile_id, bias.bias_candidate_key)

        def _do(conn: sqlite3.Connection) -> str:
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO biases (
                        id, profile_id, scope, condition_signature, preferred_policy,
                        anti_policy, rationale_summary, confidence, support_count,
                        avg_reward, recency_score, decay_rate, status, source_moment_ids,
                        created_at, updated_at, last_triggered_at, trigger_count,
                        rollback_parent_id, version, bias_candidate_key, disabled_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._bias_sql_values(bias),
                )
                self._write_bias_history(conn, bias, operation="create")
                return bias.id

            updated = PolicyBias(
                id=existing.id,
                profile_id=bias.profile_id,
                scope=bias.scope,
                condition_signature=bias.condition_signature,
                preferred_policy=bias.preferred_policy,
                anti_policy=bias.anti_policy,
                rationale_summary=bias.rationale_summary,
                confidence=bias.confidence,
                support_count=bias.support_count,
                avg_reward=bias.avg_reward,
                recency_score=bias.recency_score,
                decay_rate=bias.decay_rate,
                status=bias.status,
                source_moment_ids=bias.source_moment_ids,
                created_at=existing.created_at,
                updated_at=bias.updated_at,
                last_triggered_at=bias.last_triggered_at,
                trigger_count=bias.trigger_count,
                rollback_parent_id=bias.rollback_parent_id,
                version=max(existing.version + 1, bias.version),
                bias_candidate_key=bias.bias_candidate_key,
                status_note=bias.status_note,
            )
            conn.execute(self._bias_update_sql(), self._bias_update_values(updated))
            self._write_bias_history(conn, updated, operation="update")
            return updated.id

        return self._execute_write(_do)

    def get_bias(self, bias_id: str) -> Optional[PolicyBias]:
        row = self._conn.execute(
            "SELECT * FROM biases WHERE id = ?",
            (bias_id,),
        ).fetchone()
        return self._bias_from_row(row) if row else None

    def find_bias_by_candidate_key(
        self,
        profile_id: str,
        candidate_key: str,
    ) -> Optional[PolicyBias]:
        row = self._conn.execute(
            """
            SELECT * FROM biases
            WHERE profile_id = ? AND bias_candidate_key = ?
            LIMIT 1
            """,
            (profile_id, candidate_key),
        ).fetchone()
        return self._bias_from_row(row) if row else None

    def list_biases(
        self,
        profile_id: str,
        *,
        statuses: Optional[Iterable[str]] = None,
        scopes: Optional[Iterable[str]] = None,
        limit: int = 100,
    ) -> list[PolicyBias]:
        sql = "SELECT * FROM biases WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            sql += f" AND status IN ({placeholders})"
            params.extend(list(statuses))
        if scopes:
            placeholders = ", ".join("?" for _ in scopes)
            sql += f" AND scope IN ({placeholders})"
            params.extend(list(scopes))
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._bias_from_row(row) for row in rows]

    def delete_biases(
        self,
        profile_id: str,
        *,
        statuses: Optional[Iterable[str]] = None,
    ) -> int:
        status_list = list(statuses) if statuses else None

        def _do(conn: sqlite3.Connection) -> int:
            sql = "DELETE FROM biases WHERE profile_id = ?"
            params: list[Any] = [profile_id]
            if status_list:
                placeholders = ", ".join("?" for _ in status_list)
                sql += f" AND status IN ({placeholders})"
                params.extend(status_list)
            cur = conn.execute(sql, params)
            return cur.rowcount or 0

        return self._execute_write(_do)

    def set_bias_status(
        self,
        bias_id: str,
        *,
        status: str,
        note: Optional[str] = None,
    ) -> bool:
        def _do(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT * FROM biases WHERE id = ?",
                (bias_id,),
            ).fetchone()
            if row is None:
                return False
            existing = self._bias_from_row(row)
            updated = PolicyBias(
                id=existing.id,
                profile_id=existing.profile_id,
                scope=existing.scope,
                condition_signature=existing.condition_signature,
                preferred_policy=existing.preferred_policy,
                anti_policy=existing.anti_policy,
                rationale_summary=existing.rationale_summary,
                confidence=existing.confidence,
                support_count=existing.support_count,
                avg_reward=existing.avg_reward,
                recency_score=existing.recency_score,
                decay_rate=existing.decay_rate,
                status=status,
                source_moment_ids=existing.source_moment_ids,
                created_at=existing.created_at,
                updated_at=time.time(),
                last_triggered_at=existing.last_triggered_at,
                trigger_count=existing.trigger_count,
                rollback_parent_id=existing.rollback_parent_id,
                version=existing.version + 1,
                bias_candidate_key=existing.bias_candidate_key,
                status_note=note,
            )
            conn.execute(self._bias_update_sql(), self._bias_update_values(updated))
            self._write_bias_history(conn, updated, operation="status_change")
            return True

        return self._execute_write(_do)

    def touch_bias_trigger(self, bias_id: str) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                UPDATE biases
                SET last_triggered_at = ?, trigger_count = trigger_count + 1
                WHERE id = ?
                """,
                (time.time(), bias_id),
            )

        self._execute_write(_do)

    def list_bias_history(
        self,
        bias_id: str,
        *,
        limit: int = 20,
    ) -> list[BiasHistoryEntry]:
        rows = self._conn.execute(
            """
            SELECT * FROM bias_history
            WHERE bias_id = ?
            ORDER BY version DESC, created_at DESC
            LIMIT ?
            """,
            (bias_id, max(1, int(limit))),
        ).fetchall()
        return [self._history_from_row(row) for row in rows]

    def rollback_bias(
        self,
        bias_id: str,
        *,
        version: int,
    ) -> bool:
        def _do(conn: sqlite3.Connection) -> bool:
            target = conn.execute(
                """
                SELECT * FROM bias_history
                WHERE bias_id = ? AND version = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (bias_id, int(version)),
            ).fetchone()
            current_row = conn.execute(
                "SELECT * FROM biases WHERE id = ?",
                (bias_id,),
            ).fetchone()
            if target is None or current_row is None:
                return False

            current = self._bias_from_row(current_row)
            snapshot = _json_loads(target["snapshot"], default={}) or {}
            restored = PolicyBias(
                id=current.id,
                profile_id=snapshot.get("profile_id", current.profile_id),
                scope=snapshot.get("scope", current.scope),
                condition_signature=snapshot.get("condition_signature", current.condition_signature),
                preferred_policy=snapshot.get("preferred_policy", current.preferred_policy),
                anti_policy=snapshot.get("anti_policy"),
                rationale_summary=snapshot.get("rationale_summary", current.rationale_summary),
                confidence=float(snapshot.get("confidence", current.confidence)),
                support_count=int(snapshot.get("support_count", current.support_count)),
                avg_reward=float(snapshot.get("avg_reward", current.avg_reward)),
                recency_score=float(snapshot.get("recency_score", current.recency_score)),
                decay_rate=float(snapshot.get("decay_rate", current.decay_rate)),
                status=snapshot.get("status", current.status),
                source_moment_ids=list(snapshot.get("source_moment_ids", current.source_moment_ids)),
                created_at=float(snapshot.get("created_at", current.created_at)),
                updated_at=time.time(),
                last_triggered_at=snapshot.get("last_triggered_at", current.last_triggered_at),
                trigger_count=int(snapshot.get("trigger_count", current.trigger_count)),
                rollback_parent_id=f"{current.id}:v{current.version}",
                version=current.version + 1,
                bias_candidate_key=snapshot.get("bias_candidate_key", current.bias_candidate_key),
                status_note=snapshot.get("status_note"),
            )
            conn.execute(self._bias_update_sql(), self._bias_update_values(restored))
            self._write_bias_history(conn, restored, operation=f"rollback_to_v{version}")
            return True

        return self._execute_write(_do)

    def save_decision_trace(self, trace: DecisionTrace) -> str:
        def _do(conn: sqlite3.Connection) -> str:
            conn.execute(
                """
                INSERT OR REPLACE INTO decision_traces (
                    id, profile_id, session_id, turn_index, task_type, platform,
                    user_message_excerpt, retrieved_bias_ids, injected_bias_ids,
                    shadow_bias_ids, planner_effects, tool_weight_deltas,
                    risk_actions, evidence_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.id,
                    trace.profile_id,
                    trace.session_id,
                    trace.turn_index,
                    trace.task_type,
                    trace.platform,
                    trace.user_message_excerpt,
                    _json_dumps(trace.retrieved_bias_ids),
                    _json_dumps(trace.injected_bias_ids),
                    _json_dumps(trace.shadow_bias_ids),
                    _json_dumps(trace.planner_effects),
                    _json_dumps(trace.tool_weight_deltas),
                    _json_dumps(trace.risk_actions),
                    _json_dumps(trace.evidence_summary),
                    trace.created_at,
                ),
            )
            return trace.id

        return self._execute_write(_do)

    def get_recent_decision_traces(
        self,
        profile_id: str,
        *,
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> list[DecisionTrace]:
        sql = "SELECT * FROM decision_traces WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._trace_from_row(row) for row in rows]

    @staticmethod
    def _bias_sql_values(bias: PolicyBias) -> tuple[Any, ...]:
        return (
            bias.id,
            bias.profile_id,
            bias.scope,
            bias.condition_signature,
            bias.preferred_policy,
            bias.anti_policy,
            bias.rationale_summary,
            bias.confidence,
            bias.support_count,
            bias.avg_reward,
            bias.recency_score,
            bias.decay_rate,
            bias.status,
            _json_dumps(bias.source_moment_ids),
            bias.created_at,
            bias.updated_at,
            bias.last_triggered_at,
            bias.trigger_count,
            bias.rollback_parent_id,
            bias.version,
            bias.bias_candidate_key,
            bias.status_note,
        )

    @staticmethod
    def _bias_update_sql() -> str:
        return """
            UPDATE biases SET
                profile_id = ?, scope = ?, condition_signature = ?, preferred_policy = ?,
                anti_policy = ?, rationale_summary = ?, confidence = ?, support_count = ?,
                avg_reward = ?, recency_score = ?, decay_rate = ?, status = ?,
                source_moment_ids = ?, updated_at = ?, last_triggered_at = ?,
                trigger_count = ?, rollback_parent_id = ?, version = ?,
                bias_candidate_key = ?, disabled_reason = ?
            WHERE id = ?
        """

    @classmethod
    def _bias_update_values(cls, bias: PolicyBias) -> tuple[Any, ...]:
        return (
            bias.profile_id,
            bias.scope,
            bias.condition_signature,
            bias.preferred_policy,
            bias.anti_policy,
            bias.rationale_summary,
            bias.confidence,
            bias.support_count,
            bias.avg_reward,
            bias.recency_score,
            bias.decay_rate,
            bias.status,
            _json_dumps(bias.source_moment_ids),
            bias.updated_at,
            bias.last_triggered_at,
            bias.trigger_count,
            bias.rollback_parent_id,
            bias.version,
            bias.bias_candidate_key,
            bias.status_note,
            bias.id,
        )

    @staticmethod
    def _bias_snapshot(bias: PolicyBias) -> dict[str, Any]:
        return {
            "id": bias.id,
            "profile_id": bias.profile_id,
            "scope": bias.scope,
            "condition_signature": bias.condition_signature,
            "preferred_policy": bias.preferred_policy,
            "anti_policy": bias.anti_policy,
            "rationale_summary": bias.rationale_summary,
            "confidence": bias.confidence,
            "support_count": bias.support_count,
            "avg_reward": bias.avg_reward,
            "recency_score": bias.recency_score,
            "decay_rate": bias.decay_rate,
            "status": bias.status,
            "source_moment_ids": list(bias.source_moment_ids),
            "created_at": bias.created_at,
            "updated_at": bias.updated_at,
            "last_triggered_at": bias.last_triggered_at,
            "trigger_count": bias.trigger_count,
            "rollback_parent_id": bias.rollback_parent_id,
            "version": bias.version,
            "bias_candidate_key": bias.bias_candidate_key,
            "status_note": bias.status_note,
        }

    @classmethod
    def _write_bias_history(
        cls,
        conn: sqlite3.Connection,
        bias: PolicyBias,
        *,
        operation: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO bias_history (
                id, bias_id, profile_id, version, operation, snapshot, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("hist"),
                bias.id,
                bias.profile_id,
                bias.version,
                operation,
                _json_dumps(cls._bias_snapshot(bias)),
                time.time(),
            ),
        )

    @staticmethod
    def _moment_from_row(row: sqlite3.Row) -> PolicyMoment:
        return PolicyMoment(
            id=row["id"],
            profile_id=row["profile_id"],
            session_id=row["session_id"],
            timestamp=float(row["timestamp"]),
            task_type=row["task_type"],
            platform=row["platform"],
            context_summary=row["context_summary"],
            action_trace_summary=row["action_trace_summary"],
            tool_path=row["tool_path"],
            decision_class=row["decision_class"],
            outcome_class=row["outcome_class"],
            reward_score=float(row["reward_score"]),
            confidence_score=float(row["confidence_score"]),
            user_feedback_signal=float(row["user_feedback_signal"] or 0.0),
            error_signal=float(row["error_signal"] or 0.0),
            side_effect_level=row["side_effect_level"] or "none",
            latency_ms=row["latency_ms"],
            cost_estimate=row["cost_estimate"],
            evidence_refs=_json_loads(row["evidence_refs"], default=[]),
            extracted_tags=_json_loads(row["extracted_tags"], default=[]),
            bias_candidate_key=row["bias_candidate_key"],
        )

    @staticmethod
    def _bias_from_row(row: sqlite3.Row) -> PolicyBias:
        return PolicyBias(
            id=row["id"],
            profile_id=row["profile_id"],
            scope=row["scope"],
            condition_signature=row["condition_signature"],
            preferred_policy=row["preferred_policy"],
            anti_policy=row["anti_policy"],
            rationale_summary=row["rationale_summary"],
            confidence=float(row["confidence"]),
            support_count=int(row["support_count"]),
            avg_reward=float(row["avg_reward"]),
            recency_score=float(row["recency_score"]),
            decay_rate=float(row["decay_rate"]),
            status=row["status"],
            source_moment_ids=_json_loads(row["source_moment_ids"], default=[]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_triggered_at=row["last_triggered_at"],
            trigger_count=int(row["trigger_count"]),
            rollback_parent_id=row["rollback_parent_id"],
            version=int(row["version"]),
            bias_candidate_key=row["bias_candidate_key"],
            status_note=row["disabled_reason"] if "disabled_reason" in row.keys() else None,
        )

    @staticmethod
    def _trace_from_row(row: sqlite3.Row) -> DecisionTrace:
        return DecisionTrace(
            id=row["id"],
            profile_id=row["profile_id"],
            session_id=row["session_id"],
            turn_index=int(row["turn_index"]),
            task_type=row["task_type"],
            platform=row["platform"],
            user_message_excerpt=row["user_message_excerpt"],
            retrieved_bias_ids=_json_loads(row["retrieved_bias_ids"], default=[]),
            injected_bias_ids=_json_loads(row["injected_bias_ids"], default=[]),
            shadow_bias_ids=_json_loads(row["shadow_bias_ids"], default=[]),
            planner_effects=_json_loads(row["planner_effects"], default=[]),
            tool_weight_deltas=_json_loads(row["tool_weight_deltas"], default=[]),
            risk_actions=_json_loads(row["risk_actions"], default=[]),
            evidence_summary=_json_loads(row["evidence_summary"], default=[]),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _history_from_row(row: sqlite3.Row) -> BiasHistoryEntry:
        return BiasHistoryEntry(
            id=row["id"],
            bias_id=row["bias_id"],
            profile_id=row["profile_id"],
            version=int(row["version"]),
            operation=row["operation"],
            snapshot=_json_loads(row["snapshot"], default={}),
            created_at=float(row["created_at"]),
        )
