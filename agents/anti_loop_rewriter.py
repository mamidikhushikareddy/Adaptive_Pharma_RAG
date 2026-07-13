"""
ARAG_PHARMA — Anti-Loop Query Rewriter
"""
import hashlib
from dataclasses import dataclass
from loguru import logger
from config.settings import settings
from core.groq_client import get_groq_client


MEDRA_TERMINOLOGY = {
    "drug_interaction": [
        "pharmacokinetic interaction", "pharmacodynamic interaction",
        "CYP450 enzyme inhibition", "drug-drug interaction", "DDI",
        "concomitant use", "co-administration", "contraindicated combination",
    ],
    "adverse_event": [
        "adverse drug reaction", "ADR", "TEAE", "treatment-emergent adverse event",
        "serious adverse event", "SAE", "MedDRA preferred term",
        "post-marketing surveillance", "incidence rate",
    ],
    "clinical_trial": [
        "randomized controlled trial", "RCT", "phase III", "double-blind",
        "placebo-controlled", "primary endpoint", "overall survival",
        "progression-free survival", "NCT number", "CONSORT",
    ],
    "drug_label": [
        "prescribing information", "SmPC", "black box warning", "FDA label",
        "indications and usage", "contraindications", "warnings and precautions",
        "dosage and administration",
    ],
    "dosage_information": [
        "recommended dose", "maximum tolerated dose", "therapeutic range",
        "loading dose", "maintenance dose", "renal dose adjustment",
        "hepatic impairment", "pediatric dosing",
    ],
    "pharmacokinetics": [
        "half-life", "bioavailability", "volume of distribution",
        "clearance", "AUC", "Cmax", "Tmax", "protein binding",
        "first-pass metabolism", "steady state",
    ],
    "literature_search": [
        "systematic review", "meta-analysis", "randomized trial",
        "cohort study", "evidence-based medicine",
    ],
    "regulatory_compliance": [
        "FDA approval", "NDA", "ANDA", "IND", "EMA", "REMS",
        "post-market surveillance", "pharmacovigilance",
    ],
    "general_pharma": ["pharmaceutical", "therapeutic", "clinical evidence"],
}

REWRITE_STRATEGIES = ["synonyms", "broaden", "clinical_terms", "decompose"]


@dataclass
class RewriteResult:
    strategy: str
    original_query: str
    rewritten_query: str
    injected_terms: list[str]
    attempt_number: int


class AntiLoopRewriter:
    def __init__(self):
        self.groq = get_groq_client()
        self._query_registry: set[str] = set()

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def register_query(self, query: str) -> bool:
        h = self._hash_query(query)
        if h in self._query_registry:
            logger.warning(f"Loop detected! Query already tried: '{query[:60]}'")
            return False
        self._query_registry.add(h)
        return True

    def get_attempt_count(self) -> int:
        return len(self._query_registry)

    def is_hard_stopped(self) -> bool:
        return len(self._query_registry) >= settings.HARD_STOP_ITERATIONS

    async def rewrite(
        self, query: str, intent: str, attempt: int, failed_reason: str
    ) -> RewriteResult:
        if self.is_hard_stopped():
            logger.warning("Hard stop reached")
            return RewriteResult("hard_stop", query, query, [], attempt)

        strategy = REWRITE_STRATEGIES[(attempt - 1) % len(REWRITE_STRATEGIES)]
        terms = MEDRA_TERMINOLOGY.get(intent, MEDRA_TERMINOLOGY["general_pharma"])[:3]
        rewritten = await self._apply_strategy(query, intent, strategy, terms, failed_reason)

        if not self.register_query(rewritten):
            rewritten = await self._apply_strategy(query, intent, "broaden", terms, "duplicate")
            self.register_query(rewritten)

        logger.info(f"Rewriter [{strategy}] attempt {attempt}: '{query[:40]}' → '{rewritten[:50]}'")
        return RewriteResult(strategy, query, rewritten, terms, attempt)

    async def _apply_strategy(
        self, query: str, intent: str, strategy: str, terms: list[str], failed_reason: str
    ) -> str:
        strategy_instructions = {
            "synonyms": "Replace drug names with brand names or synonyms. Use generic if brand was used.",
            "broaden": "Remove specific constraints. Use drug class instead of specific drug names.",
            "clinical_terms": f"Add clinical/MedDRA terminology. Use these: {', '.join(terms)}",
            "decompose": "Break into a simpler sub-question focusing on just one aspect.",
        }
        instruction = strategy_instructions.get(strategy, strategy_instructions["broaden"])

        messages = [
            {
                "role": "user",
                "content": f"""Rewrite this pharmaceutical search query using the '{strategy}' strategy.

Original query: "{query}"
Intent: {intent}
Why retrieval failed: {failed_reason}
Strategy: {instruction}

Return ONLY the rewritten query (10-25 words max), no explanation, no quotes.""",
            }
        ]
        try:
            result = await self.groq.chat(
                messages=messages,
                model=settings.LLM_FAST_MODEL,
                temperature=0.3,
                max_tokens=60,
            )
            return result.strip().strip('"').strip("'")
        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            return f"{query} {terms[0] if terms else 'pharmaceutical evidence'}"
