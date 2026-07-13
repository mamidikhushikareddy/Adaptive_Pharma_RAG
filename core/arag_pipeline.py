"""
ARAG_PHARMA — Master Pipeline
"""
import time
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger

from agents.query_analyzer import QueryAnalyzer, QueryAnalysis
from agents.triple_layer_evaluator import TripleLayerEvaluator, RoutingDecision
from agents.anti_loop_rewriter import AntiLoopRewriter
from agents.conflict_detector import ConflictDetector
from agents.hallucination_checker import HallucinationChecker
from agents.quality_gate import QualityGate, REFUSAL_RESPONSE
from core.srag_module import SRAGModule
from core.freshness_tracker import freshness_tracker
from core.audit_trail import audit_trail, AuditEntry
from data.pharma_loader import PharmaDataLoader
from data.web_search_client import WebSearchClient
from config.settings import settings


@dataclass
class ARAGResponse:
    answer: str
    disclaimer: str
    is_refused: bool
    intent: str
    drug_names: list[str]
    is_high_risk: bool
    confidence_score: float
    quality_score: float
    quality_label: str
    quality_dimensions: dict
    hallucination_score: float
    hallucination_repaired: bool
    hard_caveat_added: bool
    conflicts: list[dict]
    conflict_text: str
    freshness_summary: dict
    staleness_warnings: list[str]
    sources: list[str]
    web_sources: list[str]
    requires_professional: bool
    used_web_fallback: bool
    triple_layer_scores: list[dict]
    retrieval_rounds: int
    rewrite_strategies_used: list[str]
    srag_iterations: int
    is_supported: bool
    unsupported_claims: list[str]
    audit_id: str
    evidence_chain: list[dict]
    risk_level: str
    processing_time_ms: int
    critique_log: list[dict] = field(default_factory=list)


DISCLAIMER_MAP = {
    "drug_interaction": (
        "⚕️ Drug interaction information is based on FDA-approved labeling and peer-reviewed literature. "
        "Consult a pharmacist or prescriber before combining medications."
    ),
    "ingredient_interaction": (
        "🧴 Ingredient combination information is based on published research and general evidence. "
        "Individual skin types vary. Consult a dermatologist for personalised skincare advice."
    ),
    "adverse_event": (
        "📊 Adverse event data sourced from FDA FAERS. Reports represent voluntary submissions and "
        "do not necessarily establish causality."
    ),
    "clinical_trial": (
        "🔬 Trial eligibility and status may change. Visit ClinicalTrials.gov for current information."
    ),
    "dosage_information": (
        "💊 Dosage information is from FDA-approved labeling. Doses must be determined by a licensed prescriber."
    ),
    "default": (
        "⚠️ This information is for educational purposes only and does not constitute medical advice. "
        "Always consult a qualified healthcare professional."
    ),
}


class ARAGPipeline:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.data_loader = PharmaDataLoader()
        self.triple_evaluator = TripleLayerEvaluator()
        self.conflict_detector = ConflictDetector()
        self.hallucination_checker = HallucinationChecker()
        self.quality_gate = QualityGate()
        self.srag = SRAGModule()
        self.web = WebSearchClient()
        logger.info("ARAG_PHARMA v3 pipeline initialized")

    async def run(self, query: str) -> ARAGResponse:
        start = time.time()
        run_id = audit_trail.create_run_id()
        logger.info(f"[{run_id[:8]}] Pipeline start: '{query[:70]}'")

        # ── Step 1: Query Analysis ────────────────────────────────────────────
        logger.info("[1/8] Query Analysis")
        analysis: QueryAnalysis = await self.query_analyzer.analyze(query)
        logger.info(
            f"  Intent: {analysis.intent} | Drugs: {analysis.drug_names} | "
            f"WebSearch: {analysis.use_web_search}"
        )
        if analysis.use_web_search:
            logger.info(f"  Web search reason: {analysis.web_search_reason}")

        # ── Step 2: Data Loading ──────────────────────────────────────────────
        logger.info("[2/8] Loading data sources")
        docs = await self.data_loader.load(
            query=analysis.reformulated_query,
            intent=analysis.intent,
            drug_names=analysis.drug_names,
            condition=analysis.condition,
            max_docs=settings.MAX_RETRIEVAL_ROUNDS * 3,
            use_web_search=analysis.use_web_search,
            web_search_reason=analysis.web_search_reason,
            original_query=query,
        )
        logger.info(f"  Loaded {len(docs)} docs: {list(set(d['source'] for d in docs))}")

        used_web_fallback = any(d.get("source") == settings.WEB_SOURCE_TAG for d in docs)

        # ── Step 3: Triple-Layer Evaluation ───────────────────────────────────
        logger.info("[3/8] Triple-Layer Evaluation")
        rewriter = AntiLoopRewriter()
        rewriter.register_query(analysis.reformulated_query)

        all_docs = list(docs)
        agg_score: float = 0.0
        routing = RoutingDecision.REWRITE
        rewrite_strategies = []
        retrieval_round = 1
        triple_results = []

        for attempt in range(1, settings.HARD_STOP_ITERATIONS + 1):
            eval_results, agg_score, routing = await self.triple_evaluator.evaluate_all(
                query=query, docs=all_docs
            )
            triple_results = eval_results
            logger.info(f"  Round {attempt}: score={agg_score:.2f} routing={routing.value}")

            if routing == RoutingDecision.GENERATE:
                break
            if rewriter.is_hard_stopped():
                logger.warning("  Hard stop — proceeding with available docs")
                break
            if routing == RoutingDecision.FETCH_MORE:
                extra = await self.data_loader.load(
                    query=analysis.reformulated_query,
                    intent=analysis.intent,
                    drug_names=analysis.drug_names,
                    condition=analysis.condition,
                    max_docs=4,
                    use_web_search=analysis.use_web_search,
                    original_query=query,
                )
                all_docs.extend(extra)
                retrieval_round += 1
            elif routing == RoutingDecision.REWRITE:
                rewrite = await rewriter.rewrite(
                    query=query,
                    intent=analysis.intent,
                    attempt=attempt,
                    failed_reason=f"Score {agg_score:.2f} below threshold",
                )
                rewrite_strategies.append(rewrite.strategy)
                extra = await self.data_loader.load(
                    query=rewrite.rewritten_query,
                    intent=analysis.intent,
                    drug_names=analysis.drug_names,
                    condition=analysis.condition,
                    max_docs=5,
                    use_web_search=analysis.use_web_search,
                    original_query=query,
                )
                all_docs.extend(extra)
                retrieval_round += 1

        # ── Step 4: Freshness ─────────────────────────────────────────────────
        logger.info("[4/8] Freshness Tracking")
        freshness_results, staleness_penalty, staleness_warnings = freshness_tracker.evaluate_batch(all_docs)
        freshness_summary = freshness_tracker.get_freshness_summary(freshness_results)

        # ── Step 5: Conflict Detection ────────────────────────────────────────
        logger.info("[5/8] Conflict Detection")
        conflicts, has_critical_conflict = await self.conflict_detector.detect(query, all_docs)
        conflict_text = self.conflict_detector.format_for_response(conflicts)

        # ── Step 6: Build Context ─────────────────────────────────────────────
        valid_docs = sorted(
            [r for r in triple_results if not r.discard],
            key=lambda r: r.final_score, reverse=True,
        )

        # FIX: If triple-layer discarded all pharma docs (e.g. irrelevant FDA results)
        # but we have no web docs yet, trigger a direct web search now.
        pharma_valid = [r for r in valid_docs if r.source != settings.WEB_SOURCE_TAG]
        if len(pharma_valid) == 0 and not used_web_fallback and settings.ENABLE_WEB_FALLBACK:
            logger.warning(
                "All pharma DB docs were discarded as irrelevant. "
                "Triggering emergency web search."
            )
            web_query = self.data_loader._build_web_query(query, analysis.drug_names, analysis.intent)
            emergency_web_docs = await self.web.search(web_query, max_results=5)
            if emergency_web_docs:
                all_docs.extend(emergency_web_docs)
                used_web_fallback = True
                # Re-evaluate with web docs included
                eval_results, agg_score, routing = await self.triple_evaluator.evaluate_all(
                    query=query, docs=all_docs
                )
                triple_results = eval_results
                valid_docs = sorted(
                    [r for r in triple_results if not r.discard],
                    key=lambda r: r.final_score, reverse=True,
                )
                logger.info(f"Emergency web search added {len(emergency_web_docs)} docs")

        pharma_sources = []
        web_sources = []
        context_parts = []

        for r in valid_docs[:8]:
            matching = next((d for d in all_docs if d.get("source") == r.source), None)
            if matching:
                is_web = matching.get("source") == settings.WEB_SOURCE_TAG
                provider = matching.get("provider", "")
                source_label = (
                    f"[🌐 INTERNET — {provider or r.source} | Relevance: {r.final_score:.2f}]"
                    if is_web
                    else f"[📚 {r.source} | Relevance: {r.final_score:.2f}]"
                )
                context_parts.append(
                    f"{source_label}\n{matching.get('content', '')[:1200]}"
                )
                if matching.get("url"):
                    entry = f"{r.source}: {matching['url']}"
                    if is_web:
                        web_sources.append(entry)
                    else:
                        pharma_sources.append(entry)

        # If we still have no valid docs at all, use all_docs as context of last resort
        if not context_parts and all_docs:
            logger.warning("No valid docs after scoring — using raw docs as context of last resort")
            for doc in all_docs[:5]:
                is_web = doc.get("source") == settings.WEB_SOURCE_TAG
                context_parts.append(
                    f"[{'🌐 INTERNET' if is_web else '📚 ' + doc.get('source','')}]\n"
                    f"{doc.get('content', '')[:1000]}"
                )

        context = "\n\n===\n\n".join(context_parts) if context_parts else ""
        used_sources = pharma_sources + web_sources

        # ── Step 7: SRAG Generation ───────────────────────────────────────────
        logger.info("[6/8] SRAG Generation")
        srag_result = await self.srag.process(
            query=query,
            context=context,
            analysis=analysis,
            sources=used_sources,
            used_web_fallback=used_web_fallback,
        )

        raw_response = srag_result.final_response
        if conflict_text:
            raw_response += conflict_text
        if staleness_warnings:
            raw_response += "\n\n⏰ **Data Freshness Warnings:**\n" + "\n".join(f"• {w}" for w in staleness_warnings)
        if used_web_fallback:
            raw_response += (
                "\n\n---\n🌐 **Source Note:** Some information was sourced from internet search "
                "because authoritative pharma databases returned insufficient results for this query. "
                "Internet-sourced claims are marked with [Web] and should be verified with a licensed "
                "dermatologist, pharmacist, or authoritative databases."
            )

        # ── Step 8: Hallucination Check ───────────────────────────────────────
        logger.info("[7/8] Hallucination Check")
        halluc_report = await self.hallucination_checker.check_and_repair(
            response=raw_response,
            context=context,
            query=query,
            is_high_risk=analysis.is_high_risk,
        )
        repaired_response = halluc_report.repaired_response

        # ── Step 9: Quality Gate ──────────────────────────────────────────────
        logger.info("[8/8] Quality Gate")
        quality = await self.quality_gate.assess(
            query=query,
            response=repaired_response,
            context=context,
            sources=used_sources,
            hallucination_score=halluc_report.hallucination_score,
        )

        is_refused = not quality.passed_gate
        final_response = REFUSAL_RESPONSE if is_refused else repaired_response

        confidence = self._calc_confidence(
            triple_score=agg_score,
            srag_support=srag_result.support_confidence,
            usefulness=srag_result.usefulness_score,
            halluc_score=halluc_report.hallucination_score,
            staleness_penalty=staleness_penalty,
            quality_score=quality.overall_score,
            used_web_fallback=used_web_fallback,
        )

        risk_level = self._risk_level(analysis.is_high_risk, conflicts, halluc_report, quality)
        disclaimer = DISCLAIMER_MAP.get(analysis.intent, DISCLAIMER_MAP["default"])
        requires_professional = (
            analysis.is_high_risk or has_critical_conflict
            or halluc_report.hard_caveat_added or risk_level in ("HIGH", "CRITICAL")
        )
        elapsed_ms = int((time.time() - start) * 1000)

        evidence_chain = audit_trail.build_evidence_chain(
            final_response,
            [{"source": r.source, "content": r.content_preview, "url": r.url} for r in valid_docs],
        )
        entry = AuditEntry(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            intent=analysis.intent,
            drug_names=analysis.drug_names,
            sources_queried=list(set(d["source"] for d in all_docs)),
            docs_retrieved=len(all_docs),
            docs_used=len(valid_docs),
            fallback_triggered=retrieval_round > 1 or used_web_fallback,
            retrieval_rounds=retrieval_round,
            rewrite_strategies_used=rewrite_strategies,
            relevance_scores=[{"source": r.source, "score": r.final_score} for r in triple_results],
            conflict_detected=bool(conflicts),
            conflicts=[{"topic": c.topic, "severity": c.severity} for c in conflicts],
            staleness_flags=staleness_warnings,
            srag_iterations=srag_result.iterations,
            hallucination_flags=halluc_report.hallucinated_claims,
            hallucination_repaired=bool(halluc_report.repair_actions),
            quality_score=quality.overall_score,
            confidence_score=confidence,
            claim_evidence=[{"claim": e.claim, "source": e.source_name} for e in evidence_chain],
            response_preview=final_response[:200],
            risk_level=risk_level,
            processing_time_ms=elapsed_ms,
        )
        audit_trail.log(entry)
        logger.info(
            f"[{entry.audit_signature}] Done in {elapsed_ms}ms | "
            f"Q: {quality.quality_label} | Conf: {confidence:.0%} | Web: {used_web_fallback}"
        )

        return ARAGResponse(
            answer=final_response,
            disclaimer=disclaimer,
            is_refused=is_refused,
            intent=analysis.intent,
            drug_names=analysis.drug_names,
            is_high_risk=analysis.is_high_risk,
            confidence_score=confidence,
            quality_score=quality.overall_score,
            quality_label=quality.quality_label,
            quality_dimensions={
                "factual_accuracy": quality.factual_accuracy,
                "completeness": quality.completeness,
                "source_attribution": quality.source_attribution,
                "safety_compliance": quality.safety_compliance,
                "clarity": quality.clarity,
                "actionability": quality.actionability,
            },
            hallucination_score=halluc_report.hallucination_score,
            hallucination_repaired=bool(halluc_report.repair_actions),
            hard_caveat_added=halluc_report.hard_caveat_added,
            conflicts=[{"topic": c.topic, "severity": c.severity, "resolution": c.resolution} for c in conflicts],
            conflict_text=conflict_text,
            freshness_summary=freshness_summary,
            staleness_warnings=staleness_warnings,
            sources=list(dict.fromkeys(pharma_sources))[:8],
            web_sources=list(dict.fromkeys(web_sources))[:6],
            requires_professional=requires_professional,
            used_web_fallback=used_web_fallback,
            triple_layer_scores=[
                {
                    "source": r.source,
                    "final_score": r.final_score,
                    "semantic": r.layer1_semantic.score,
                    "trust": r.layer2_trust.score,
                    "consistency": r.layer3_consistency.score,
                    "discarded": r.discard,
                }
                for r in triple_results
            ],
            retrieval_rounds=retrieval_round,
            rewrite_strategies_used=rewrite_strategies,
            srag_iterations=srag_result.iterations,
            is_supported=srag_result.is_supported,
            unsupported_claims=srag_result.unsupported_claims,
            audit_id=entry.audit_signature,
            evidence_chain=[
                {"claim": e.claim, "source": e.source_name, "url": e.source_url, "confidence": e.confidence}
                for e in evidence_chain
            ],
            risk_level=risk_level,
            processing_time_ms=elapsed_ms,
            critique_log=srag_result.critique_log,
        )

    def _calc_confidence(
        self, triple_score, srag_support, usefulness,
        halluc_score, staleness_penalty, quality_score, used_web_fallback=False,
    ) -> float:
        raw = (
            triple_score * 0.25 + srag_support * 0.20 + usefulness * 0.15
            + (1 - halluc_score) * 0.20 + quality_score * 0.20
        )
        adjusted = raw - staleness_penalty
        if used_web_fallback:
            adjusted -= 0.05  # Slight penalty for relying on internet sources
        return round(min(1.0, max(0.0, adjusted)), 3)

    def _risk_level(self, is_high_risk, conflicts, halluc_report, quality) -> str:
        if is_high_risk and halluc_report.hallucination_score > 0.5:
            return "CRITICAL"
        if is_high_risk or any(c.severity == "HIGH" for c in conflicts):
            return "HIGH"
        if halluc_report.hard_caveat_added or quality.quality_label == "POOR":
            return "MEDIUM"
        return "LOW"
