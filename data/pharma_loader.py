"""
ARAG_PHARMA — Multi-Source Pharma Data Loader

"""
import asyncio
from loguru import logger
from config.settings import settings
from data.fda_client import FDAClient
from data.pubmed_client import PubMedClient
from data.clinicaltrials_client import ClinicalTrialsClient
from data.web_search_client import WebSearchClient


# Intents that benefit from web search even when pharma DBs return results
WEB_PREFERRED_INTENTS = {
    "ingredient_interaction",
    "general_pharma",
    "literature_search",
}

# Intents where FDA label search is reliable and should be the primary source
FDA_RELIABLE_INTENTS = {
    "drug_label",
    "dosage_information",
    "regulatory_compliance",
    "adverse_event",
    "drug_interaction",
}


class PharmaDataLoader:
    """
    Multi-source pharma data loader with intelligent web fallback.
    - For prescription drug queries: FDA + PubMed primary, web as fallback
    - For cosmetic/OTC/nutraceutical queries: PubMed primary + web always
    - Web docs are clearly tagged so LLM knows to cite them as internet sources
    """

    APPROVED_SOURCE_MAP = {
        "drug_interaction":        ["fda_label", "pubmed"],
        "ingredient_interaction":  ["pubmed"],           # No FDA label for cosmetics
        "adverse_event":           ["fda_faers", "pubmed"],
        "drug_label":              ["fda_label"],
        "dosage_information":      ["fda_label", "pubmed"],
        "clinical_trial":          ["clinical_trials", "pubmed"],
        "literature_search":       ["pubmed"],
        "pharmacokinetics":        ["pubmed", "fda_label"],
        "regulatory_compliance":   ["fda_label"],
        "general_pharma":          ["pubmed"],            # Skip FDA for generic queries
    }

    def __init__(self):
        self.fda = FDAClient()
        self.pubmed = PubMedClient()
        self.ct = ClinicalTrialsClient()
        self.web = WebSearchClient()

    async def load(
        self,
        query: str,
        intent: str,
        drug_names: list[str],
        condition: str = "",
        max_docs: int = 10,
        use_web_search: bool = False,      # From QueryAnalysis
        web_search_reason: str = "",
        original_query: str = "",
    ) -> list[dict]:
        """
        Load documents from approved sources based on intent.

        Key logic:
        - If use_web_search=True (set by QueryAnalyzer for OTC/cosmetic queries),
          ALWAYS include web search regardless of pharma DB results.
        - If intent is ingredient_interaction, skip FDA label search (returns garbage).
        - Web fallback triggers when RELEVANT docs < WEB_FALLBACK_MIN_DOCS.
        """
        sources_to_use = self.APPROVED_SOURCE_MAP.get(intent, ["pubmed"])
        tasks = []

        # --- Build PubMed query that's specific to the ingredients ---
        pubmed_query = self._build_pubmed_query(query, drug_names, intent)

        for source in sources_to_use:
            if source == "fda_label":
                # FIX: Skip FDA label search for cosmetic/OTC ingredients —
                # it returns unrelated branded products and pollutes the context.
                if use_web_search or intent in ("ingredient_interaction",):
                    logger.info(f"Skipping FDA label search for OTC/cosmetic query (intent={intent})")
                    continue
                targets = drug_names[:2] if drug_names else []
                # Only search FDA if we have actual prescription drug names
                prescription_targets = [
                    d for d in targets
                    if d.lower() not in {
                        "niacinamide", "nicotinamide", "salicylic acid",
                        "hyaluronic acid", "sodium hyaluronate", "retinol",
                        "vitamin c", "ascorbic acid", "vitamin e", "tocopherol",
                        "glycolic acid", "lactic acid", "azelaic acid", "ceramide",
                    }
                ]
                for drug in prescription_targets[:2]:
                    tasks.append(self.fda.search_drug_label(drug, limit=2))

            elif source == "fda_faers":
                targets = drug_names[:2] if drug_names else []
                for drug in targets:
                    tasks.append(self.fda.search_adverse_events(drug, limit=4))

            elif source == "pubmed":
                # Use ingredient-specific query for PubMed
                tasks.append(self.pubmed.search_and_fetch(pubmed_query, max_results=5))

            elif source == "clinical_trials":
                tasks.append(self.ct.search(
                    condition=condition or query,
                    intervention=drug_names[0] if drug_names else "",
                    max_results=4,
                ))

        # Gather all pharma DB results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        docs = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Data source error: {r}")
                continue
            if isinstance(r, list):
                docs.extend(r)

        # Validate approved sources
        approved = [
            d for d in docs
            if d.get("source", "") in settings.APPROVED_SOURCES
        ]
        if len(approved) < len(docs):
            logger.warning(f"Rejected {len(docs) - len(approved)} docs from unapproved sources")

        logger.info(f"Loaded {len(approved)} approved docs | sources: {sources_to_use} | intent: {intent}")

        # ── Web Search Logic ──────────────────────────────────────────────────
        # Condition 1: QueryAnalyzer flagged this as a web-preferred query
        # Condition 2: Standard fallback — approved docs below minimum threshold
        # Condition 3: Web-preferred intents always get supplemental web results
        relevant_count = self._count_relevant_docs(approved, drug_names)

        should_web_search = (
            (settings.ENABLE_WEB_FALLBACK and use_web_search)
            or (settings.ENABLE_WEB_FALLBACK and relevant_count < settings.WEB_FALLBACK_MIN_DOCS)
            or (settings.ENABLE_WEB_FALLBACK and intent in WEB_PREFERRED_INTENTS and len(approved) < 3)
        )

        if should_web_search:
            reason = web_search_reason or (
                f"Only {relevant_count} relevant approved docs found" if relevant_count < settings.WEB_FALLBACK_MIN_DOCS
                else f"Intent '{intent}' benefits from web supplementation"
            )
            logger.info(f"🌐 Triggering web search: {reason}")

            # Build a targeted web search query
            web_query = self._build_web_query(
                original_query or query, drug_names, intent
            )
            web_docs = await self.web.search(web_query, max_results=5)
            if web_docs:
                logger.info(f"Web search returned {len(web_docs)} docs for: '{web_query[:60]}'")
                approved.extend(web_docs)
            else:
                logger.warning(f"Web search returned no results for: '{web_query[:60]}'")

        return approved[:max_docs]

    def _count_relevant_docs(self, docs: list[dict], drug_names: list[str]) -> int:
        """
        Count docs that are actually relevant to the drug/ingredient names.
        A doc is relevant if its content mentions at least one of the drug names.
        This prevents irrelevant FDA docs from falsely satisfying the quota.
        """
        if not drug_names:
            return len(docs)
        count = 0
        for doc in docs:
            content_lower = doc.get("content", "").lower()
            if any(name.lower() in content_lower for name in drug_names):
                count += 1
        return count

    def _build_pubmed_query(self, query: str, drug_names: list[str], intent: str) -> str:
        """Build an optimized PubMed search query."""
        if drug_names:
            # For multi-ingredient queries, search for the combination
            ingredient_terms = " AND ".join(f'"{name}"' for name in drug_names[:3])
            if intent in ("ingredient_interaction", "drug_interaction"):
                return f"{ingredient_terms} skin topical effects"
            return f"{ingredient_terms} {query[:50]}"
        return query

    def _build_web_query(self, query: str, drug_names: list[str], intent: str) -> str:
        """Build an optimized web search query."""
        if drug_names and len(drug_names) >= 2:
            names_str = " ".join(drug_names[:3])
            if intent in ("ingredient_interaction",):
                return f"{names_str} combined effects skin benefits interactions"
            if "effect" in query.lower() or "combined" in query.lower():
                return f"{names_str} combined effects benefits interactions"
            return f"{names_str} {query[:60]}"
        return f"{query} pharmaceutical effects benefits"
