"""
ARAG_PHARMA — Query Analyzer
FIX: Recognise cosmetic/nutraceutical/skincare ingredients as valid pharmaceutical
     compounds. The old code only matched a hardcoded list of prescription drugs,
     so niacinamide, hyaluronic acid, salicylic acid etc. were never extracted.
FIX: Detect when a query is about OTC/cosmetic ingredients and set
     intent=ingredient_interaction + force web fallback via use_web_search flag.
FIX: Added use_web_search flag to QueryAnalysis so the pipeline can always
     trigger web search for topics where FDA/PubMed coverage is sparse.
"""
import re
from dataclasses import dataclass, field
from loguru import logger
from config.settings import settings
from core.groq_client import get_groq_client


@dataclass
class QueryAnalysis:
    original_query: str
    intent: str
    drug_names: list[str]
    condition: str
    reformulated_query: str
    injected_terms: list[str]
    is_high_risk: bool
    confidence: float
    # NEW: signals that web search should always be used (e.g. cosmetics/nutraceuticals)
    use_web_search: bool = False
    # NEW: human-readable reason why web search is preferred
    web_search_reason: str = ""


# Ingredients/compounds that are OTC, cosmetic or nutraceutical and rarely in
# FDA prescription drug labels — web search gives much better results for these.
OTC_COSMETIC_INGREDIENTS = {
    "niacinamide", "nicotinamide", "salicylic acid", "hyaluronic acid",
    "sodium hyaluronate", "retinol", "retinoid", "tretinoin", "vitamin c",
    "ascorbic acid", "vitamin e", "tocopherol", "glycolic acid", "lactic acid",
    "azelaic acid", "kojic acid", "alpha arbutin", "arbutin", "tranexamic acid",
    "benzoyl peroxide", "zinc oxide", "titanium dioxide", "ceramide", "peptide",
    "collagen", "squalane", "niacinamide", "aha", "bha", "pha", "centella",
    "green tea extract", "caffeine", "resveratrol", "ferulic acid",
    "panthenol", "allantoin", "zinc", "sulfur", "tea tree", "mandelic acid",
    "polyglutamic acid", "beta glucan", "snail mucin", "bakuchiol",
    "turmeric", "curcumin", "coenzyme q10", "ubiquinone", "licorice extract",
    "mushroom extract", "oat extract", "aloe vera", "probiotics", "prebiotics",
    "omega 3", "fish oil", "vitamin d", "magnesium", "iron", "folate",
    "biotin", "melatonin", "ashwagandha", "valerian", "chamomile",
    "glucosamine", "chondroitin", "msm", "collagen peptides",
}

# Topics where PubMed covers well but FDA labels are poor — prefer PubMed + web
PUBMED_PREFERRED_INTENTS = {
    "ingredient_interaction", "cosmetic_ingredient", "nutraceutical",
    "supplement_interaction", "literature_search", "general_pharma",
}


class QueryAnalyzer:
    def __init__(self):
        self.groq = get_groq_client()

    async def analyze(self, query: str) -> QueryAnalysis:
        try:
            return await self._llm_analyze(query)
        except Exception as e:
            logger.error(f"LLM analysis failed, using fallback: {e}")
            return self._fallback(query)

    async def _llm_analyze(self, query: str) -> QueryAnalysis:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a pharmaceutical and cosmetic ingredient query analyzer. "
                    "You understand both prescription drugs AND cosmetic/OTC/nutraceutical ingredients. "
                    "Always respond with ONLY a valid JSON object, no extra text."
                ),
            },
            {
                "role": "user",
                "content": f"""Analyze this pharmaceutical/cosmetic query and return JSON.

Query: "{query}"

IMPORTANT NOTES:
- drug_names should include ALL chemical/ingredient names mentioned, including cosmetic ingredients
  (e.g. niacinamide, salicylic acid, hyaluronic acid, retinol, vitamin C, etc.)
- If the query is about skincare/cosmetic/OTC/supplement ingredients, set intent to "ingredient_interaction"
  or "general_pharma" and set "use_web_search" to true since FDA drug labels don't cover these well
- reformulated_query should be a search-engine-optimized query

Return this exact JSON structure:
{{
  "intent": "<one of: drug_interaction|ingredient_interaction|clinical_trial|adverse_event|drug_label|literature_search|dosage_information|pharmacokinetics|regulatory_compliance|general_pharma>",
  "drug_names": ["<ALL ingredient/compound names mentioned, even cosmetic ones>"],
  "condition": "<skin condition, medical condition, or empty string>",
  "reformulated_query": "<precise optimized query for database and web search>",
  "use_web_search": <true if cosmetic/OTC/supplement ingredients involved, false otherwise>,
  "web_search_reason": "<why web search helps, or empty string>",
  "confidence": 0.85
}}""",
            },
        ]

        data = await self.groq.chat_json(
            messages=messages,
            model=settings.LLM_FAST_MODEL,
            temperature=0.1,
            max_tokens=400,
            fallback={
                "intent": "general_pharma",
                "drug_names": [],
                "condition": "",
                "reformulated_query": query,
                "use_web_search": False,
                "web_search_reason": "",
                "confidence": 0.5,
            },
        )

        drug_names = data.get("drug_names", [])
        if not drug_names:
            # Fallback: extract ingredient names directly from query using regex
            drug_names = self._extract_ingredients_from_query(query)

        is_high_risk = any(
            d.lower() in " ".join(settings.HIGH_RISK_DRUGS)
            for d in drug_names
        )

        # Override use_web_search if any OTC/cosmetic ingredient is detected
        use_web_search = data.get("use_web_search", False)
        web_search_reason = data.get("web_search_reason", "")
        detected_otc = [
            d for d in drug_names
            if any(otc in d.lower() for otc in OTC_COSMETIC_INGREDIENTS)
        ]
        if detected_otc and not use_web_search:
            use_web_search = True
            web_search_reason = (
                f"Cosmetic/OTC ingredients detected ({', '.join(detected_otc)}). "
                f"FDA drug labels have sparse coverage — web search provides better evidence."
            )

        intent = data.get("intent", "general_pharma")

        from agents.anti_loop_rewriter import MEDRA_TERMINOLOGY
        injected = MEDRA_TERMINOLOGY.get(intent, MEDRA_TERMINOLOGY.get("general_pharma", []))[:2]

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            drug_names=drug_names,
            condition=data.get("condition", ""),
            reformulated_query=data.get("reformulated_query", query),
            injected_terms=injected,
            is_high_risk=is_high_risk,
            confidence=float(data.get("confidence", 0.7)),
            use_web_search=use_web_search,
            web_search_reason=web_search_reason,
        )

    def _extract_ingredients_from_query(self, query: str) -> list[str]:
        """
        Regex-based ingredient extraction as a safety net when LLM returns empty list.
        Looks for known OTC/cosmetic ingredient names in the query text.
        """
        q_lower = query.lower()
        found = []
        for ingredient in OTC_COSMETIC_INGREDIENTS:
            if ingredient in q_lower:
                # Preserve original casing from query
                idx = q_lower.find(ingredient)
                found.append(query[idx:idx + len(ingredient)])
        # Also extract capitalised words that look like ingredient names
        caps_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        for match in caps_pattern.finditer(query):
            word = match.group()
            if len(word) > 3 and word not in found:
                found.append(word)
        return found[:6]  # cap at 6

    def _fallback(self, query: str) -> QueryAnalysis:
        q = query.lower()

        # Detect cosmetic/OTC queries first
        otc_matches = [ing for ing in OTC_COSMETIC_INGREDIENTS if ing in q]
        if otc_matches:
            intent = "ingredient_interaction"
            use_web_search = True
            web_reason = f"OTC/cosmetic ingredients detected: {', '.join(otc_matches[:3])}"
            drug_names = [query[q.find(m):q.find(m) + len(m)].strip() for m in otc_matches]
        elif any(w in q for w in ["interact", "combination", "combined", "together", "mix"]):
            intent = "drug_interaction"
            use_web_search = False
            web_reason = ""
            drug_names = []
        elif any(w in q for w in ["trial", "clinical", "study", "recruit"]):
            intent = "clinical_trial"
            use_web_search = False
            web_reason = ""
            drug_names = []
        elif any(w in q for w in ["side effect", "adverse", "reaction"]):
            intent = "adverse_event"
            use_web_search = False
            web_reason = ""
            drug_names = []
        elif any(w in q for w in ["dose", "dosage", "how much", "mg"]):
            intent = "dosage_information"
            use_web_search = False
            web_reason = ""
            drug_names = []
        elif any(w in q for w in ["pharmacokinetics", "half-life", "bioavailability"]):
            intent = "pharmacokinetics"
            use_web_search = False
            web_reason = ""
            drug_names = []
        elif any(w in q for w in ["fda", "approval", "regulatory", "nda"]):
            intent = "regulatory_compliance"
            use_web_search = False
            web_reason = ""
            drug_names = []
        else:
            intent = "general_pharma"
            use_web_search = False
            web_reason = ""
            drug_names = []

        # Also scan for common prescription drugs
        if not otc_matches:
            common_drugs = [
                "warfarin", "aspirin", "metformin", "lisinopril", "methotrexate",
                "pembrolizumab", "nivolumab", "atorvastatin", "ibuprofen",
                "acetaminophen", "insulin", "digoxin", "amoxicillin", "omeprazole",
                "prednisone", "heparin", "vancomycin", "tacrolimus", "cyclosporine",
            ]
            drug_names = [d for d in common_drugs if d in q]

        is_high_risk = any(d.lower() in settings.HIGH_RISK_DRUGS for d in drug_names)

        from agents.anti_loop_rewriter import MEDRA_TERMINOLOGY
        injected = MEDRA_TERMINOLOGY.get(intent, MEDRA_TERMINOLOGY.get("general_pharma", []))[:2]

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            drug_names=drug_names,
            condition="",
            reformulated_query=query,
            injected_terms=injected,
            is_high_risk=is_high_risk,
            confidence=0.5,
            use_web_search=use_web_search,
            web_search_reason=web_reason,
        )
