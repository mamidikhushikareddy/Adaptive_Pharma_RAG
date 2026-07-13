"""
ARAG_PHARMA — Audit Trail System
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from loguru import logger
from config.settings import settings


@dataclass
class ClaimEvidence:
    """Links a specific claim in the response to its source document."""
    claim: str                  # The specific claim made
    source_name: str            # e.g., "FDA Drug Label"
    source_url: str             # Direct URL
    source_excerpt: str         # Excerpt that supports the claim
    confidence: float           # How strongly the source supports this claim
    is_verified: bool           # Was this verified by triple-layer eval?


@dataclass
class AuditEntry:
    """Complete audit record for one pipeline run."""
    run_id: str
    timestamp: str
    query: str
    intent: str
    drug_names: list[str]

    # Retrieval audit
    sources_queried: list[str]
    docs_retrieved: int
    docs_used: int
    fallback_triggered: bool
    retrieval_rounds: int
    rewrite_strategies_used: list[str]

    # Evaluation audit
    relevance_scores: list[dict]
    conflict_detected: bool
    conflicts: list[dict]
    staleness_flags: list[str]

    # Generation audit
    srag_iterations: int
    hallucination_flags: list[str]
    hallucination_repaired: bool
    quality_score: float
    confidence_score: float

    # Evidence chain
    claim_evidence: list[dict]

    # Final
    response_preview: str
    risk_level: str
    processing_time_ms: int

    # Fix #9: Every output has a complete audit ID for traceability
    audit_signature: str = ""

    def __post_init__(self):
        if not self.audit_signature:
            self.audit_signature = f"ARAG-{self.run_id[:8].upper()}"


class AuditTrail:
    """
    Fix #9: Full audit trail system.
    Logs every pipeline run with complete evidence chain.
    """

    def __init__(self):
        self.enabled = settings.ENABLE_AUDIT_TRAIL
        log_path = Path(settings.AUDIT_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

    def create_run_id(self) -> str:
        return str(uuid.uuid4())

    def log(self, entry: AuditEntry) -> None:
        """Append audit entry to JSONL log file."""
        if not self.enabled:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            logger.debug(f"Audit logged: {entry.audit_signature}")
        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")

    def build_evidence_chain(
        self,
        response: str,
        scored_docs: list[dict],
    ) -> list[ClaimEvidence]:
        """
        Build claim → source evidence chain.
        Extracts sentences from response and matches to source documents.
        """
        evidence = []
        # Extract sentences from response (simple split)
        sentences = [s.strip() for s in response.replace("\n", " ").split(".") if len(s.strip()) > 30]

        for sentence in sentences[:10]:  # Top 10 claims
            # Find best matching source
            best_score = 0.0
            best_doc = None
            for doc in scored_docs:
                content = doc.get("content", "")
                # Simple overlap scoring
                words_in_claim = set(sentence.lower().split())
                words_in_doc = set(content.lower().split())
                overlap = len(words_in_claim & words_in_doc) / max(len(words_in_claim), 1)
                if overlap > best_score:
                    best_score = overlap
                    best_doc = doc

            if best_doc and best_score > 0.15:
                # Extract relevant excerpt from doc
                doc_content = best_doc.get("content", "")
                excerpt_start = doc_content.lower().find(sentence.split()[0].lower() if sentence.split() else "")
                if excerpt_start >= 0:
                    excerpt = doc_content[max(0, excerpt_start-50):excerpt_start+200]
                else:
                    excerpt = doc_content[:200]

                evidence.append(ClaimEvidence(
                    claim=sentence + ".",
                    source_name=best_doc.get("source", "Unknown"),
                    source_url=best_doc.get("url", ""),
                    source_excerpt=excerpt[:300],
                    confidence=round(best_score, 3),
                    is_verified=best_score > 0.3,
                ))

        return evidence

    def get_recent_runs(self, n: int = 10) -> list[dict]:
        """Read last N audit entries."""
        if not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
            entries = [json.loads(line) for line in lines[-n:] if line]
            return entries
        except Exception as e:
            logger.warning(f"Audit read failed: {e}")
            return []


# Global audit trail instance
audit_trail = AuditTrail()
