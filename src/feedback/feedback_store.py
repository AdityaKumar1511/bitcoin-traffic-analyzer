"""
Analyst Feedback Store Module.

Persists and manages analyst feedback (e.g., CONFIRMED, FALSE_POSITIVE)
to support alert triaging and recalibration of composite risk scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logging_config import get_logger
from src.utils.paths import get_project_root

logger = get_logger(__name__)


class FeedbackStore:
    """
    SQLite-backed persistent store for investigator/analyst alert evaluations.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize database connection and schema.
        """
        if db_path is None:
            db_path = get_project_root() / "data" / "processed" / "analyst_feedback.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    status TEXT NOT NULL,  -- PENDING, CONFIRMED, FALSE_POSITIVE
                    notes TEXT,
                    analyst_id TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def set_feedback(
        self,
        entity_id: str,
        status: str,
        entity_type: str = "wallet",
        notes: str = "",
        analyst_id: str = "analyst_1",
    ) -> None:
        """
        Record or update analyst feedback for an entity.
        """
        clean_status = status.upper().strip()
        if clean_status not in ("PENDING", "CONFIRMED", "FALSE_POSITIVE"):
            raise ValueError(f"Invalid status: {status}. Must be PENDING, CONFIRMED, or FALSE_POSITIVE.")

        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (entity_id, entity_type, status, notes, analyst_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    status=excluded.status,
                    notes=excluded.notes,
                    analyst_id=excluded.analyst_id,
                    updated_at=excluded.updated_at
                """,
                (entity_id, entity_type, clean_status, notes, analyst_id, now_str),
            )
            conn.commit()
        logger.info("Feedback recorded for entity %s: %s", entity_id, clean_status)

    def get_feedback(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve feedback record for an entity.
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT entity_id, entity_type, status, notes, analyst_id, updated_at FROM feedback WHERE entity_id = ?",
                (entity_id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "entity_id": row[0],
                    "entity_type": row[1],
                    "status": row[2],
                    "notes": row[3],
                    "analyst_id": row[4],
                    "updated_at": row[5],
                }
        return None

    def get_all_feedback(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all feedback records as a dictionary keyed by entity_id.
        """
        results: Dict[str, Dict[str, Any]] = {}
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT entity_id, entity_type, status, notes, analyst_id, updated_at FROM feedback")
            for row in cur.fetchall():
                results[row[0]] = {
                    "entity_id": row[0],
                    "entity_type": row[1],
                    "status": row[2],
                    "notes": row[3],
                    "analyst_id": row[4],
                    "updated_at": row[5],
                }
        return results

    def recalibrate_scores(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalibrate composite risk scores based on accumulated analyst feedback.

        - Confirmed True Positives receive risk score amplification (+20% up to 1.0).
        - Confirmed False Positives receive aggressive risk score suppression (-90% down to 0.1x).
        - Updates risk levels accordingly.
        """
        if alerts_df.empty:
            return alerts_df

        feedback_map = self.get_all_feedback()
        if not feedback_map:
            return alerts_df

        recalibrated = alerts_df.copy()
        for entity_id, fb in feedback_map.items():
            if entity_id in recalibrated.index:
                status = fb.get("status")
                notes = fb.get("notes", "")
                recalibrated.loc[entity_id, "analyst_status"] = status
                recalibrated.loc[entity_id, "analyst_notes"] = notes

                orig_score = float(recalibrated.loc[entity_id, "composite_risk_score"])
                if status == "CONFIRMED":
                    new_score = round(min(1.0, orig_score * 1.20), 4)
                elif status == "FALSE_POSITIVE":
                    new_score = round(orig_score * 0.10, 4)
                else:
                    new_score = orig_score

                recalibrated.loc[entity_id, "composite_risk_score"] = new_score
                if new_score >= 0.75:
                    recalibrated.loc[entity_id, "risk_level"] = "CRITICAL"
                elif new_score >= 0.50:
                    recalibrated.loc[entity_id, "risk_level"] = "HIGH"
                elif new_score >= 0.30:
                    recalibrated.loc[entity_id, "risk_level"] = "MEDIUM"
                else:
                    recalibrated.loc[entity_id, "risk_level"] = "LOW"

        logger.info("Recalibrated risk scores for %d reviewed entities.", len(feedback_map))
        return recalibrated.sort_values(by="composite_risk_score", ascending=False)


