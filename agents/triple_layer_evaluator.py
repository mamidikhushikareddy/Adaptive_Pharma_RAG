"""
ARAG_PHARMA — Triple-Layer Relevance Evaluator
"""
import json
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from config.settings import settings
from core.groq_client import get_groq_client


class RoutingDecision(str, Enum):
    GENERATE = "GENERATE"
    FETCH_MORE = "FETCH_MORE"
    REWRITE = "REWRITE"


@dataclass
class LayerScore:
    layer: str
    score: float
    weight: float
    reason: str
    details: dict


@dataclass
class TripleLayerResult:
    document_id: str
    source: str
    source_type: str
    content_preview: str
    url: str
    layer1_semantic: LayerScore
    layer2_trust: LayerScore
    layer3_consistency: LayerScore
    final_score: float
    routing: RoutingDecision
    key_facts: list[str]
    discard: bool
    discard_reason: str


class TripleLayerEvaluator:
    WEIGHTS = {"semantic": 0.50, "trust": 0.25, "consistency": 0.25}

    def __init__(self):
        self.groq = get_groq_client()

    async def evaluate_all(
        self, query: str, docs: list[dict]
    ) -> tuple[list[TripleLayerResult], float, RoutingDecision]:
        if not docs:
            return [], 0.0, RoutingDecision.REWRITE

        results = []
        for i, doc in enumerate(docs):
            result = await self._evaluate_one(query, doc, i, docs)
            results.append(result)

        valid = [r for r in results if not r.discard]
        aggregate = sum(r.final_score for r in valid) / len(valid) if valid else 0.0
        routing = self._route(aggregate)

        logger.info(
            f"Triple-layer: {len(valid)}/{len(results)} valid | "
            f"Score={aggregate:.2f} | Routing={routing.value}"
        )
        return results, round(aggregate, 3), routing

    async def _evaluate_one(
        self, query: str, doc: dict, doc_idx: int, all_docs: list[dict]
    ) -> TripleLayerResult:
        l1 = await self._layer1_semantic(query, doc)
        l2 = self._layer2_trust(doc)
        l3 = await self._layer3_consistency(query, doc, doc_idx, all_docs)

        final = (
            l1.score * self.WEIGHTS["semantic"]
            + l2.score * self.WEIGHTS["trust"]
            + l3.score * self.WEIGHTS["consistency"]
        )
        final = round(final, 3)
        key_facts = l1.details.get("key_facts", [])
        discard = final < 0.20

        return TripleLayerResult(
            document_id=f"doc_{doc_idx}",
            source=doc.get("source", "Unknown"),
            source_type=doc.get("source_type", "unknown"),
            content_preview=doc.get("content", "")[:200],
            url=doc.get("url", ""),
            layer1_semantic=l1,
            layer2_trust=l2,
            layer3_consistency=l3,
            final_score=final,
            routing=self._route(final),
            key_facts=key_facts,
            discard=discard,
            discard_reason=f"Score {final:.2f} below 0.20" if discard else "",
        )

    async def _layer1_semantic(self, query: str, doc: dict) -> LayerScore:
        messages = [
            {
                "role": "user",
                "content": f"""Rate the pharmaceutical relevance of this document to the query.
Return ONLY JSON.

QUERY: {query}
SOURCE: {doc.get('source', 'Unknown')} ({doc.get('source_type', 'unknown')})
CONTENT: {doc.get('content', '')[:800]}

JSON format:
{{"score": 0.85, "reason": "one sentence", "key_facts": ["fact1", "fact2", "fact3"]}}

Score: 1.0=perfectly relevant, 0.7=mostly relevant, 0.4=partial, 0.1=irrelevant""",
            }
        ]
        data = await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_FAST_MODEL,
            temperature=0.1,
            max_tokens=250,
            fallback={"score": 0.5, "reason": "eval failed", "key_facts": []},
        )
        return LayerScore(
            layer="semantic_llm",
            score=float(data.get("score", 0.5)),
            weight=self.WEIGHTS["semantic"],
            reason=data.get("reason", ""),
            details={"key_facts": data.get("key_facts", [])},
        )

    def _layer2_trust(self, doc: dict) -> LayerScore:
        source = doc.get("source", "Unknown")
        trust = settings.SOURCE_TRUST_SCORES.get(source, 0.65)
        if doc.get("source_type") == "drug_label":
            trust = min(1.0, trust + 0.02)
        return LayerScore(
            layer="rule_based_trust",
            score=trust,
            weight=self.WEIGHTS["trust"],
            reason=f"Institutional trust score for '{source}': {trust:.0%}",
            details={"source": source, "trust_score": trust},
        )

    async def _layer3_consistency(
        self, query: str, doc: dict, doc_idx: int, all_docs: list[dict]
    ) -> LayerScore:
        if len(all_docs) < 2:
            return LayerScore(
                "cross_doc_consistency", 0.80, self.WEIGHTS["consistency"],
                "Single doc — no cross-check", {}
            )

        others = [d for i, d in enumerate(all_docs) if i != doc_idx][:3]
        other_summaries = "\n".join(
            f"[{d.get('source')}]: {d.get('content', '')[:250]}"
            for d in others
        )
        messages = [
            {
                "role": "user",
                "content": f"""Does this pharmaceutical document agree with other sources?
Return ONLY JSON.

QUERY: {query}
DOCUMENT ({doc.get('source', 'Unknown')}): {doc.get('content', '')[:400]}
OTHER SOURCES: {other_summaries}

JSON format:
{{"consistency_score": 0.8, "assessment": "CONSISTENT", "reason": "one sentence"}}

assessment options: CONSISTENT, MOSTLY_CONSISTENT, MIXED, INCONSISTENT
score: 1.0=fully consistent, 0.5=mixed, 0.0=contradicts""",
            }
        ]
        data = await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_FAST_MODEL,
            temperature=0.1,
            max_tokens=200,
            fallback={"consistency_score": 0.70, "assessment": "MIXED", "reason": "eval failed"},
        )
        return LayerScore(
            layer="cross_doc_consistency",
            score=float(data.get("consistency_score", 0.7)),
            weight=self.WEIGHTS["consistency"],
            reason=data.get("reason", ""),
            details={"assessment": data.get("assessment", "MIXED")},
        )

    def _route(self, score: float) -> RoutingDecision:
        if score >= settings.CONFIDENCE_GENERATE_THRESHOLD:
            return RoutingDecision.GENERATE
        elif score >= settings.CONFIDENCE_REWRITE_THRESHOLD:
            return RoutingDecision.FETCH_MORE
        else:
            return RoutingDecision.REWRITE
