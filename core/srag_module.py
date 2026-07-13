"""
ARAG_PHARMA — Self-RAG (SRAG) Module
"""
from dataclasses import dataclass
from loguru import logger
from config.settings import settings
from core.groq_client import get_groq_client


@dataclass
class SRAGResult:
    final_response: str
    retrieve_needed: bool
    is_supported: bool
    support_confidence: float
    unsupported_claims: list[str]
    usefulness_score: float
    iterations: int
    critique_log: list[dict]


class SRAGModule:
    def __init__(self):
        self.groq = get_groq_client()

    async def process(
        self, query: str, context: str, analysis, sources: list[str],
        used_web_fallback: bool = False,
    ) -> SRAGResult:
        if not settings.ENABLE_SELF_REFLECTION:
            response = await self._generate(query, context, analysis, used_web_fallback=used_web_fallback)
            return SRAGResult(response, True, True, 0.9, [], 0.85, 1, [])

        critique_log = [{"step": "Retrieve", "decision": True, "reason": "Pharma always requires retrieval"}]

        response = await self._generate(query, context, analysis, used_web_fallback=used_web_fallback)
        iteration = 1
        sup_data = {"is_supported": True, "confidence": 0.7, "unsupported_claims": []}
        use_data = {"is_useful": True, "score": 0.75, "issues": []}

        while iteration <= settings.SRAG_MAX_ITERATIONS:
            sup_data = await self._check_support(response, context, query)
            critique_log.append({"step": "IsSUP", "iteration": iteration, **sup_data})

            use_data = await self._check_usefulness(query, response)
            critique_log.append({"step": "IsUSE", "iteration": iteration, **use_data})

            if sup_data.get("is_supported") and use_data.get("is_useful"):
                break

            if iteration < settings.SRAG_MAX_ITERATIONS:
                feedback = self._build_feedback(sup_data, use_data)
                response = await self._generate(
                    query, context, analysis, feedback,
                    used_web_fallback=used_web_fallback,
                )
            iteration += 1

        return SRAGResult(
            final_response=response,
            retrieve_needed=True,
            is_supported=sup_data.get("is_supported", True),
            support_confidence=sup_data.get("confidence", 0.7),
            unsupported_claims=sup_data.get("unsupported_claims", []),
            usefulness_score=use_data.get("score", 0.7),
            iterations=iteration,
            critique_log=critique_log,
        )

    async def _generate(
        self, query: str, context: str, analysis,
        feedback: str = "", used_web_fallback: bool = False,
    ) -> str:
        has_context = bool(context and context.strip() and context != "No sufficient context retrieved.")

        # FIX: Build source attribution instruction based on what we have
        if used_web_fallback and has_context:
            source_instruction = """SOURCE ATTRIBUTION RULES:
- The context contains both [📚 PHARMA DB] and [🌐 INTERNET] labelled sections.
- After each key claim, cite the source: [FDA Drug Label], [PubMed: title], [Web: provider], or [Knowledge Base]
- Claims from internet sources MUST include: "(Internet source — verify independently)"
- Claims from your training knowledge MUST include: "[Knowledge Base — not from retrieved data]"
"""
        elif has_context:
            source_instruction = """SOURCE ATTRIBUTION RULES:
- Cite the specific source after each key claim: [FDA Drug Label], [PubMed PMID:xxxxx], [ClinicalTrials NCT-xxxxx]
- If a claim comes from your training knowledge (not retrieved context), mark it: [Knowledge Base]
"""
        else:
            source_instruction = """SOURCE ATTRIBUTION RULES:
- No database documents were retrieved for this query.
- All claims come from your training knowledge — mark each with: [Knowledge Base]
- Be transparent about this limitation and recommend the user verify with authoritative sources.
"""

        # FIX: Changed from "ONLY the provided context" to a balanced prompt that
        # allows knowledge-based answers when context is irrelevant/empty
        if has_context:
            context_instruction = f"""RETRIEVED EVIDENCE:
{context}

Use the above retrieved context as your PRIMARY source of information.
If the context directly addresses the query, base your answer on it.
If the context is irrelevant or insufficient, supplement with your pharmaceutical/
dermatological training knowledge — but clearly label those claims as [Knowledge Base]."""
        else:
            context_instruction = """No relevant documents were retrieved from the pharma databases.
Answer using your pharmaceutical and dermatological training knowledge.
Be transparent that this answer is based on general knowledge, not retrieved documents.
Recommend the user verify with PubMed, dermatologist, or authoritative databases."""

        messages = [
            {"role": "system", "content": settings.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Answer this pharmaceutical/cosmetic query.

QUERY: {query}
INTENT: {analysis.intent}
INGREDIENTS/DRUGS: {', '.join(analysis.drug_names) if analysis.drug_names else 'Not specified'}
HIGH-RISK: {'YES — extra caution required' if analysis.is_high_risk else 'No'}

{context_instruction}

{source_instruction}

RESPONSE FORMAT:
- Start with a direct answer to the question
- For multi-ingredient queries: discuss each ingredient individually, then their combination
- Include: benefits, interactions, precautions, recommended usage order (if skincare)
- State overall confidence: HIGH / MODERATE / LOW
- End with a brief professional consultation note

{"REVISION FEEDBACK — address these issues:\n" + feedback if feedback else ""}

Provide a complete, helpful, specific answer. Do NOT say "the context doesn't mention these ingredients" — if context is irrelevant, use your knowledge and label it clearly.""",
            },
        ]
        return await self.groq.chat(
            messages=messages,
            model=settings.LLM_MODEL,
            temperature=0.15,
            max_tokens=1800,
        )

    async def _check_support(self, response: str, context: str, query: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"""Evaluate if this pharmaceutical response is well-supported.
Return ONLY JSON.

Note: Claims labelled [Knowledge Base] are acceptable even without context support.
Claims about specific drug interactions, dosages or trial data SHOULD have context support.

CONTEXT: {context[:1500] if context else "No context available"}
RESPONSE: {response[:900]}

JSON format:
{{"is_supported": true, "confidence": 0.85, "unsupported_claims": ["claim if any"]}}""",
            }
        ]
        return await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_FAST_MODEL,
            temperature=0.1,
            max_tokens=300,
            fallback={"is_supported": True, "confidence": 0.6, "unsupported_claims": []},
        )

    async def _check_usefulness(self, query: str, response: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"""Is this pharmaceutical/cosmetic AI response useful and complete?
Return ONLY JSON.

QUERY: {query}
RESPONSE: {response[:800]}

A useful response: directly answers the question, gives specific information,
doesn't just say "consult a professional" without providing any actual information.

JSON format:
{{"is_useful": true, "score": 0.8, "issues": ["issue if any"]}}""",
            }
        ]
        return await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_FAST_MODEL,
            temperature=0.1,
            max_tokens=200,
            fallback={"is_useful": True, "score": 0.7, "issues": []},
        )

    def _build_feedback(self, sup_data: dict, use_data: dict) -> str:
        parts = []
        if not sup_data.get("is_supported"):
            parts.append(f"UNSUPPORTED CLAIMS — remove or clearly mark as [Knowledge Base]: {sup_data.get('unsupported_claims', [])}")
        if not use_data.get("is_useful"):
            parts.append(f"USEFULNESS ISSUES — make the answer more specific and actionable: {use_data.get('issues', [])}")
        return "\n".join(parts)
