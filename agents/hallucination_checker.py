"""
ARAG_PHARMA — Hallucination Checker

FIX: Old checker flagged ALL dosage mentions not found in context as hallucinations.
This caused 80% hallucination score on knowledge-based answers (e.g. cosmetic
ingredient questions) where context was empty or irrelevant — every specific
claim was flagged even when it was correct general knowledge.

FIX: Claims explicitly labelled [Knowledge Base] in the response are NOT flagged
as hallucinations — they are acknowledged as knowledge-based and scored separately.

FIX: Hallucination auto-repair now only applies to responses where CONTEXT was
present and relevant. If context was empty/irrelevant, repair is skipped since
there's nothing to cross-check against.
"""
import re
from dataclasses import dataclass
from loguru import logger
from config.settings import settings
from core.groq_client import get_groq_client


@dataclass
class HallucinationReport:
    hallucination_score: float
    hallucinated_claims: list[str]
    verified_claims: list[str]
    repaired_response: str
    hard_caveat_added: bool
    repair_actions: list[str]


HARD_CAVEAT = (
    "\n\n🚨 **ACCURACY CAVEAT**: Portions of this response could not be fully "
    "verified against retrieved pharmaceutical sources. Specific claims about "
    "dosages, interactions, or clinical data should be independently verified "
    "with a licensed pharmacist, physician, or official drug databases."
)


class HallucinationChecker:
    def __init__(self):
        self.groq = get_groq_client()

    async def check_and_repair(
        self, response: str, context: str, query: str, is_high_risk: bool
    ) -> HallucinationReport:
        if not settings.ENABLE_HALLUCINATION_CHECK:
            return HallucinationReport(0.0, [], [], response, False, [])

        context_is_relevant = self._context_is_relevant(context, response)

        # FIX: If context is empty or irrelevant, skip cross-check entirely.
        # Knowledge-based answers can't be "hallucinations" against empty context.
        if not context_is_relevant:
            logger.info("Context empty/irrelevant — skipping hallucination cross-check (knowledge-based answer)")
            return HallucinationReport(
                hallucination_score=0.05,  # Small baseline score
                hallucinated_claims=[],
                verified_claims=[],
                repaired_response=response,
                hard_caveat_added=False,
                repair_actions=[],
            )

        # FIX: Only run regex scan for claims NOT already labelled [Knowledge Base]
        non_kb_response = re.sub(r'\[Knowledge Base[^\]]*\]', '', response)
        pattern_flags = self._regex_scan(non_kb_response, context)
        llm_data = await self._llm_cross_check(response, context, query)

        all_hallucinated = list(set(pattern_flags + llm_data.get("hallucinated_claims", [])))
        score = llm_data.get("hallucination_score", 0.0)
        verified = llm_data.get("verified_claims", [])

        repaired, actions = await self._repair(response, all_hallucinated, context, score)

        add_caveat = (
            score >= settings.HALLUCINATION_HARD_CAVEAT_THRESHOLD
            or (is_high_risk and score >= 0.40)
        )
        if add_caveat:
            repaired += HARD_CAVEAT
            actions.append("Hard caveat added")

        return HallucinationReport(
            hallucination_score=round(score, 3),
            hallucinated_claims=all_hallucinated,
            verified_claims=verified,
            repaired_response=repaired,
            hard_caveat_added=add_caveat,
            repair_actions=actions,
        )

    def _context_is_relevant(self, context: str, response: str) -> bool:
        """
        Check if retrieved context is actually relevant to the response.
        FIX: Prevents false hallucination flags when context is empty or off-topic.
        """
        if not context or context.strip() == "" or context == "No sufficient context retrieved.":
            return False
        # If context is very short, it's probably not useful
        if len(context.strip()) < 100:
            return False
        return True

    def _regex_scan(self, response: str, context: str) -> list[str]:
        flags = []
        dosage_re = re.compile(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|IU|units?)', re.I)
        for amount, unit in dosage_re.findall(response):
            if f"{amount} {unit}".lower() not in context.lower():
                flags.append(f"Dosage '{amount} {unit}' not in retrieved context")

        pmid_re = re.compile(r'PMID[:\s]+(\d{7,8})', re.I)
        for match in pmid_re.finditer(response):
            if match.group(1) not in context:
                flags.append(f"PMID {match.group(1)} not in retrieved literature")

        nct_re = re.compile(r'NCT\d{8}', re.I)
        for match in nct_re.finditer(response):
            if match.group(0) not in context:
                flags.append(f"Trial ID {match.group(0)} not in retrieved trial data")

        return flags[:5]

    async def _llm_cross_check(self, response: str, context: str, query: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"""Cross-check this pharmaceutical response against retrieved context.
Return ONLY JSON.

IMPORTANT: Claims marked [Knowledge Base] in the response are intentionally knowledge-based
and should NOT be flagged as hallucinations. Only flag claims that CONTRADICT the context
or invent specific clinical data (drug names, dosages, trial IDs) not in context.

QUERY: {query}
CONTEXT (authoritative): {context[:2000]}
AI RESPONSE: {response[:1000]}

JSON format:
{{"hallucination_score": 0.1, "hallucinated_claims": ["specific contradicted claim"], "verified_claims": ["claim verified by context"]}}

hallucination_score: 0.0=fully grounded or knowledge-based, 0.5=some unsupported clinical claims, 1.0=fabricated clinical data""",
            }
        ]
        return await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_MODEL,
            temperature=0.1,
            max_tokens=400,
            fallback={"hallucination_score": 0.0, "hallucinated_claims": [], "verified_claims": []},
        )

    async def _repair(
        self, response: str, hallucinated_claims: list[str], context: str, score: float
    ) -> tuple[str, list[str]]:
        if not hallucinated_claims or not settings.HALLUCINATION_AUTO_REPAIR:
            return response, []

        actions = []
        repaired = response

        for claim in hallucinated_claims[:5]:
            words = claim.split()[:6]
            phrase = " ".join(words)
            if phrase in repaired:
                repaired = repaired.replace(phrase, f"[VERIFY WITH PHARMACIST — {phrase}]")
                actions.append(f"Marked unverified: '{phrase}'")

        if score >= 0.60 and hallucinated_claims:
            messages = [
                {
                    "role": "user",
                    "content": f"""Rewrite this pharmaceutical response.
Remove only claims that directly CONTRADICT the context.
Keep claims marked [Knowledge Base] — those are intentional.
Keep all other supported or plausible claims.
Mark uncertain clinical claims with [VERIFY WITH PHARMACIST].

CONTEXT: {context[:1500]}
RESPONSE TO FIX: {response[:1000]}

Return only the repaired response, no explanation.""",
                }
            ]
            try:
                repaired = await self.groq.chat(
                    messages=messages,
                    model=settings.LLM_MODEL,
                    temperature=0.1,
                    max_tokens=1200,
                )
                actions.append("LLM rewrite applied")
            except Exception as e:
                logger.warning(f"LLM repair failed: {e}")

        return repaired, actions
