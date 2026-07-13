"""
ARAG_PHARMA — Source Conflict Detector
"""
import json
from dataclasses import dataclass
from loguru import logger
from config.settings import settings
from core.groq_client import get_groq_client


@dataclass
class ConflictRecord:
    topic: str
    source_a: str
    claim_a: str
    source_b: str
    claim_b: str
    severity: str
    resolution: str
    trust_winner: str


class ConflictDetector:
    def __init__(self):
        self.groq = get_groq_client()

    async def detect(
        self, query: str, docs: list[dict]
    ) -> tuple[list[ConflictRecord], bool]:
        if not settings.ENABLE_CONFLICT_DETECTION or len(docs) < 2:
            return [], False

        conflicts = await self._llm_conflict_check(query, docs[:6])
        resolved = [self._resolve_conflict(c) for c in conflicts]
        has_critical = any(c.severity in ("HIGH", "CRITICAL") for c in resolved)

        if resolved:
            logger.warning(f"Conflicts found: {len(resolved)}")

        return resolved, has_critical

    async def _llm_conflict_check(self, query: str, docs: list[dict]) -> list[ConflictRecord]:
        doc_summaries = "\n\n".join(
            f"[Doc {i+1} — {doc.get('source', 'Unknown')}]:\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(docs)
        )

        messages = [
            {
                "role": "user",
                "content": f"""Analyze these pharmaceutical documents for DIRECT contradictions relevant to: "{query}"

Documents:
{doc_summaries}

Find cases where documents DIRECTLY CONTRADICT each other on dosages, interactions, contraindications, or safety data.

Return ONLY JSON:
{{"conflicts": [{{"topic": "what they disagree on", "source_a": "Doc N source", "claim_a": "what Doc N says", "source_b": "Doc M source", "claim_b": "what Doc M says", "severity": "LOW|MEDIUM|HIGH|CRITICAL"}}]}}

If no conflicts: {{"conflicts": []}}""",
            }
        ]

        data = await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_MODEL,
            temperature=0.1,
            max_tokens=600,
            fallback={"conflicts": []},
        )

        records = []
        for c in data.get("conflicts", []):
            records.append(ConflictRecord(
                topic=c.get("topic", ""),
                source_a=c.get("source_a", ""),
                claim_a=c.get("claim_a", ""),
                source_b=c.get("source_b", ""),
                claim_b=c.get("claim_b", ""),
                severity=c.get("severity", "MEDIUM"),
                resolution="",
                trust_winner="",
            ))
        return records

    def _resolve_conflict(self, conflict: ConflictRecord) -> ConflictRecord:
        trust_a = settings.SOURCE_TRUST_SCORES.get(conflict.source_a, 0.70)
        trust_b = settings.SOURCE_TRUST_SCORES.get(conflict.source_b, 0.70)

        if trust_a > trust_b:
            winner = conflict.source_a
        elif trust_b > trust_a:
            winner = conflict.source_b
        else:
            winner = "Tie — equal trust"

        conflict.trust_winner = winner
        conflict.resolution = (
            f"Prioritizing {winner} by source trust score. "
            f"{conflict.source_a}={trust_a:.0%} vs {conflict.source_b}={trust_b:.0%}. "
            "Both claims shown for transparency."
        )
        return conflict

    def format_for_response(self, conflicts: list[ConflictRecord]) -> str:
        if not conflicts:
            return ""
        lines = ["\n\n⚔️ **SOURCE CONFLICTS DETECTED** — Both claims shown for transparency:\n"]
        for i, c in enumerate(conflicts, 1):
            lines.append(
                f"{i}. **{c.topic}** [{c.severity}]\n"
                f"   • {c.source_a}: {c.claim_a}\n"
                f"   • {c.source_b}: {c.claim_b}\n"
                f"   → {c.resolution}\n"
            )
        return "\n".join(lines)
