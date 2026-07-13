"""
ARAG_PHARMA — Quality Gate 
"""
from dataclasses import dataclass
from loguru import logger
from config.settings import settings
from core.groq_client import get_groq_client


@dataclass
class QualityAssessment:
    overall_score: float
    factual_accuracy: float
    completeness: float
    source_attribution: float
    safety_compliance: float
    clarity: float
    actionability: float
    passed_gate: bool
    gate_reason: str
    improvement_suggestions: list[str]
    quality_label: str


QUALITY_LABELS = {
    (0.80, 1.01): "EXCELLENT",
    (0.65, 0.80): "GOOD",
    (0.40, 0.65): "ACCEPTABLE",
    (0.20, 0.40): "POOR",
    (0.00, 0.20): "REJECTED",
}

REFUSAL_RESPONSE = (
    "I was unable to generate a sufficiently accurate response to your query. "
    "The available pharmaceutical databases don't have adequate information on this specific topic.\n\n"
    "**Recommended Resources:**\n"
    "• Consult a licensed pharmacist or clinical pharmacologist\n"
    "• Search PubMed directly: https://pubmed.ncbi.nlm.nih.gov\n"
    "• Check FDA drug databases: https://www.accessdata.fda.gov\n"
    "• Review ClinicalTrials.gov for trial-specific questions\n\n"
    "For urgent medical questions, contact a healthcare provider immediately."
)


class QualityGate:
    def __init__(self):
        self.groq = get_groq_client()

    async def assess(
        self,
        query: str,
        response: str,
        context: str,
        sources: list[str],
        hallucination_score: float,
    ) -> QualityAssessment:
        if not settings.ENABLE_QUALITY_GATE:
            return self._default_pass()

        scores = await self._llm_assess(query, response, context, sources)

        scores["factual_accuracy"] = max(
            0.0, scores.get("factual_accuracy", 0.7) - hallucination_score * 0.5
        )

        weights = {
            "factual_accuracy": 0.30,
            "completeness": 0.20,
            "source_attribution": 0.20,
            "safety_compliance": 0.15,
            "clarity": 0.10,
            "actionability": 0.05,
        }
        overall = sum(scores.get(k, 0.5) * w for k, w in weights.items())
        overall = round(min(1.0, max(0.0, overall)), 3)

        label = "REJECTED"
        for (low, high), lbl in QUALITY_LABELS.items():
            if low <= overall < high:
                label = lbl
                break

        passed = overall >= settings.QUALITY_GATE_MIN_SCORE
        gate_reason = (
            f"Score {overall:.0%} {'meets' if passed else 'below'} "
            f"minimum {settings.QUALITY_GATE_MIN_SCORE:.0%}"
        )

        if not passed:
            logger.warning(f"Quality gate FAILED: {overall:.0%}")

        return QualityAssessment(
            overall_score=overall,
            factual_accuracy=round(scores.get("factual_accuracy", 0.5), 3),
            completeness=round(scores.get("completeness", 0.5), 3),
            source_attribution=round(scores.get("source_attribution", 0.5), 3),
            safety_compliance=round(scores.get("safety_compliance", 0.5), 3),
            clarity=round(scores.get("clarity", 0.5), 3),
            actionability=round(scores.get("actionability", 0.5), 3),
            passed_gate=passed,
            gate_reason=gate_reason,
            improvement_suggestions=scores.get("improvements", []),
            quality_label=label,
        )

    async def _llm_assess(
        self, query: str, response: str, context: str, sources: list[str]
    ) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"""Assess the quality of this pharmaceutical AI response.
Return ONLY JSON.

QUERY: {query}
SOURCES USED: {', '.join(sources[:5]) if sources else 'None'}
RESPONSE: {response[:1000]}

Score each 0.0-1.0:
- factual_accuracy: Are facts correct and supported by context?
- completeness: Does it fully answer the query?
- source_attribution: Are sources cited?
- safety_compliance: Appropriate disclaimers, avoids medical advice?
- clarity: Clear and structured?
- actionability: Useful and actionable?

JSON format:
{{"factual_accuracy": 0.8, "completeness": 0.7, "source_attribution": 0.9, "safety_compliance": 0.95, "clarity": 0.85, "actionability": 0.75, "improvements": ["suggestion1"]}}""",
            }
        ]
        return await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_MODEL,
            temperature=0.1,
            max_tokens=400,
            fallback={
                "factual_accuracy": 0.60, "completeness": 0.60, "source_attribution": 0.50,
                "safety_compliance": 0.70, "clarity": 0.65, "actionability": 0.55,
                "improvements": [],
            },
        )

    def _default_pass(self) -> QualityAssessment:
        return QualityAssessment(
            overall_score=0.75, factual_accuracy=0.75, completeness=0.75,
            source_attribution=0.70, safety_compliance=0.80, clarity=0.75,
            actionability=0.70, passed_gate=True, gate_reason="Quality gate disabled",
            improvement_suggestions=[], quality_label="GOOD",
        )
