"""
ARAG_PHARMA — Data Freshness Tracker
"""
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from loguru import logger
from config.settings import settings


@dataclass
class FreshnessResult:
    """Freshness evaluation for a single document."""
    source: str
    retrieved_at: datetime
    is_stale: bool
    hours_old: float
    staleness_penalty: float   # Confidence reduction
    freshness_label: str       # FRESH | AGING | STALE | VERY_STALE


class FreshnessTracker:
    """
    Fix #8: Tracks data freshness for all retrieved documents.

    Staleness levels:
    - FRESH:      < 6h  → no penalty
    - AGING:      6-24h → 2% penalty
    - STALE:      24-48h → 10% penalty
    - VERY_STALE: > 48h → 20% penalty + explicit warning in response
    """

    STALENESS_THRESHOLDS = {
        "FRESH": 6,       # hours
        "AGING": 24,
        "STALE": 48,
        "VERY_STALE": float("inf"),
    }

    PENALTIES = {
        "FRESH": 0.00,
        "AGING": 0.02,
        "STALE": 0.10,
        "VERY_STALE": 0.20,
    }

    def evaluate(self, source: str, retrieved_at: datetime) -> FreshnessResult:
        """Evaluate freshness of a single document."""
        now = datetime.now(timezone.utc)
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)

        hours_old = (now - retrieved_at).total_seconds() / 3600

        # Determine label
        if hours_old < self.STALENESS_THRESHOLDS["FRESH"]:
            label = "FRESH"
        elif hours_old < self.STALENESS_THRESHOLDS["AGING"]:
            label = "AGING"
        elif hours_old < self.STALENESS_THRESHOLDS["STALE"]:
            label = "STALE"
        else:
            label = "VERY_STALE"

        penalty = self.PENALTIES[label]
        is_stale = label in ("STALE", "VERY_STALE")

        if is_stale:
            logger.warning(f"Stale data from {source}: {hours_old:.1f}h old ({label})")

        return FreshnessResult(
            source=source,
            retrieved_at=retrieved_at,
            is_stale=is_stale,
            hours_old=round(hours_old, 1),
            staleness_penalty=penalty,
            freshness_label=label,
        )

    def evaluate_batch(
        self, docs: list[dict]
    ) -> tuple[list[FreshnessResult], float, list[str]]:
        """
        Evaluate all documents.
        Returns: (freshness_results, total_penalty, staleness_warnings)
        """
        results = []
        total_penalty = 0.0
        warnings = []

        for doc in docs:
            retrieved_at = doc.get("retrieved_at", datetime.now(timezone.utc))
            if isinstance(retrieved_at, str):
                try:
                    retrieved_at = datetime.fromisoformat(retrieved_at)
                except ValueError:
                    retrieved_at = datetime.now(timezone.utc)

            result = self.evaluate(doc.get("source", "Unknown"), retrieved_at)
            results.append(result)
            total_penalty += result.staleness_penalty

            if result.is_stale:
                warnings.append(
                    f"⏰ {result.source} data is {result.freshness_label} "
                    f"({result.hours_old:.0f}h old) — verify with current sources"
                )

        return results, min(total_penalty, 0.30), warnings  # Cap total penalty at 30%

    def get_freshness_summary(self, results: list[FreshnessResult]) -> dict:
        """Summary stats for UI display."""
        if not results:
            return {"fresh": 0, "aging": 0, "stale": 0, "very_stale": 0}
        counts = {"FRESH": 0, "AGING": 0, "STALE": 0, "VERY_STALE": 0}
        for r in results:
            counts[r.freshness_label] = counts.get(r.freshness_label, 0) + 1
        return {k.lower(): v for k, v in counts.items()}


freshness_tracker = FreshnessTracker()
