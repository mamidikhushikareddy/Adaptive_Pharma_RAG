"""
ARAG_PHARMA — Full Test Suite
"""
import pytest
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone, timedelta


# ── Fix #1: Triple-Layer Evaluator ──────────────────────────────────────────

class TestTripleLayerEvaluator:
    def test_layer2_trust_fda(self):
        from agents.triple_layer_evaluator import TripleLayerEvaluator
        ev = TripleLayerEvaluator()
        doc = {"source": "FDA", "source_type": "drug_label", "content": "test"}
        result = ev._layer2_trust(doc)
        assert result.score >= 0.95

    def test_layer2_trust_pubmed(self):
        from agents.triple_layer_evaluator import TripleLayerEvaluator
        ev = TripleLayerEvaluator()
        doc = {"source": "PubMed", "source_type": "literature", "content": "test"}
        result = ev._layer2_trust(doc)
        assert result.score >= 0.85

    def test_routing_generate(self):
        from agents.triple_layer_evaluator import TripleLayerEvaluator, RoutingDecision
        ev = TripleLayerEvaluator()
        assert ev._route(0.80) == RoutingDecision.GENERATE

    def test_routing_fetch_more(self):
        from agents.triple_layer_evaluator import TripleLayerEvaluator, RoutingDecision
        ev = TripleLayerEvaluator()
        assert ev._route(0.42) == RoutingDecision.FETCH_MORE

    def test_routing_rewrite(self):
        from agents.triple_layer_evaluator import TripleLayerEvaluator, RoutingDecision
        ev = TripleLayerEvaluator()
        assert ev._route(0.20) == RoutingDecision.REWRITE


# ── Fix #2: Continuous Confidence ───────────────────────────────────────────

class TestConfidenceThresholds:
    def test_thresholds_order(self):
        from config.settings import settings
        assert settings.CONFIDENCE_REWRITE_THRESHOLD < settings.CONFIDENCE_FETCH_MORE_THRESHOLD
        assert settings.CONFIDENCE_FETCH_MORE_THRESHOLD < settings.CONFIDENCE_GENERATE_THRESHOLD


# ── Fix #3: Anti-Loop Rewriter ───────────────────────────────────────────────

class TestAntiLoopRewriter:
    def test_query_registry_detects_loop(self):
        from agents.anti_loop_rewriter import AntiLoopRewriter
        rw = AntiLoopRewriter()
        rw.register_query("warfarin interactions")
        result = rw.register_query("warfarin interactions")  # Same query
        assert result == False

    def test_query_registry_new_query(self):
        from agents.anti_loop_rewriter import AntiLoopRewriter
        rw = AntiLoopRewriter()
        rw.register_query("warfarin interactions")
        result = rw.register_query("aspirin contraindications")  # Different query
        assert result == True

    def test_hard_stop(self):
        from agents.anti_loop_rewriter import AntiLoopRewriter
        from config.settings import settings
        rw = AntiLoopRewriter()
        for i in range(settings.HARD_STOP_ITERATIONS):
            rw.register_query(f"unique query {i}")
        assert rw.is_hard_stopped() == True

    def test_not_hard_stopped_initially(self):
        from agents.anti_loop_rewriter import AntiLoopRewriter
        rw = AntiLoopRewriter()
        assert rw.is_hard_stopped() == False

    def test_strategy_cycles(self):
        from agents.anti_loop_rewriter import REWRITE_STRATEGIES
        assert len(REWRITE_STRATEGIES) == 4

    def test_medra_terminology_9_intents(self):
        from agents.anti_loop_rewriter import MEDRA_TERMINOLOGY
        from config.settings import settings
        for intent in settings.INTENT_CATEGORIES:
            assert intent in MEDRA_TERMINOLOGY, f"Missing MedDRA terms for intent: {intent}"


# ── Fix #4: Approved Sources ─────────────────────────────────────────────────

class TestApprovedSources:
    def test_all_intents_have_approved_sources(self):
        from data.pharma_loader import PharmaDataLoader
        from config.settings import settings
        loader = PharmaDataLoader()
        for intent in settings.INTENT_CATEGORIES:
            assert intent in loader.APPROVED_SOURCE_MAP

    def test_web_fallback_disabled(self):
        from config.settings import settings
        assert settings.ENABLE_WEB_FALLBACK == False


# ── Fix #5: Hallucination Checker ────────────────────────────────────────────

class TestHallucinationChecker:
    def test_dosage_not_in_context_flagged(self):
        from agents.hallucination_checker import HallucinationChecker
        checker = HallucinationChecker()
        response = "The patient should take 500 mg of warfarin daily."
        context = "Warfarin is an anticoagulant. Doses vary widely."
        flags = checker._regex_scan(response, context)
        assert len(flags) > 0

    def test_dosage_in_context_not_flagged(self):
        from agents.hallucination_checker import HallucinationChecker
        checker = HallucinationChecker()
        response = "The standard dose is 5 mg daily."
        context = "Typical dosing: 5 mg daily for most patients."
        flags = checker._regex_scan(response, context)
        assert len(flags) == 0

    def test_fabricated_pmid_flagged(self):
        from agents.hallucination_checker import HallucinationChecker
        checker = HallucinationChecker()
        response = "According to PMID 12345678, warfarin requires monitoring."
        context = "Warfarin requires INR monitoring."
        flags = checker._regex_scan(response, context)
        assert any("PMID" in f for f in flags)


# ── Fix #6: Conflict Detector ────────────────────────────────────────────────

class TestConflictDetector:
    def test_resolution_fda_wins_over_unknown(self):
        from agents.conflict_detector import ConflictDetector, ConflictRecord
        detector = ConflictDetector()
        conflict = ConflictRecord(
            topic="Dosage",
            source_a="FDA",
            claim_a="5mg daily",
            source_b="Unknown Blog",
            claim_b="10mg daily",
            severity="HIGH",
            resolution="",
            trust_winner="",
        )
        resolved = detector._resolve_conflict(conflict)
        assert resolved.trust_winner == "FDA"

    def test_conflict_format(self):
        from agents.conflict_detector import ConflictDetector, ConflictRecord
        detector = ConflictDetector()
        conflict = ConflictRecord("Dosage", "FDA", "5mg", "PubMed", "10mg", "MEDIUM", "FDA preferred", "FDA")
        text = detector.format_for_response([conflict])
        assert "Dosage" in text
        assert "MEDIUM" in text


# ── Fix #7: Query Analyzer ───────────────────────────────────────────────────

class TestQueryAnalyzer:
    def test_9_intent_categories(self):
        from config.settings import settings
        assert len(settings.INTENT_CATEGORIES) == 9

    def test_fallback_drug_interaction(self):
        from agents.query_analyzer import QueryAnalyzer
        qa = QueryAnalyzer()
        result = qa._fallback("does warfarin interact with aspirin?")
        assert result.intent == "drug_interaction"
        assert "warfarin" in result.drug_names

    def test_fallback_clinical_trial(self):
        from agents.query_analyzer import QueryAnalyzer
        qa = QueryAnalyzer()
        result = qa._fallback("active clinical trials for lung cancer")
        assert result.intent == "clinical_trial"

    def test_fallback_pharmacokinetics(self):
        from agents.query_analyzer import QueryAnalyzer
        qa = QueryAnalyzer()
        result = qa._fallback("what is the half-life of vancomycin?")
        assert result.intent == "pharmacokinetics"

    def test_high_risk_drug_flagged(self):
        from agents.query_analyzer import QueryAnalyzer
        qa = QueryAnalyzer()
        result = qa._fallback("warfarin dosage for atrial fibrillation")
        assert result.is_high_risk == True

    def test_medra_terms_injected(self):
        from agents.query_analyzer import QueryAnalyzer
        qa = QueryAnalyzer()
        result = qa._fallback("warfarin aspirin interaction")
        assert len(result.injected_terms) > 0


# ── Fix #8: Freshness Tracker ────────────────────────────────────────────────

class TestFreshnessTracker:
    def test_fresh_no_penalty(self):
        from core.freshness_tracker import FreshnessTracker
        tracker = FreshnessTracker()
        now = datetime.now(timezone.utc)
        result = tracker.evaluate("FDA", now)
        assert result.freshness_label == "FRESH"
        assert result.staleness_penalty == 0.0

    def test_stale_has_penalty(self):
        from core.freshness_tracker import FreshnessTracker
        tracker = FreshnessTracker()
        old = datetime.now(timezone.utc) - timedelta(hours=50)
        result = tracker.evaluate("FDA", old)
        assert result.freshness_label == "VERY_STALE"
        assert result.staleness_penalty > 0

    def test_aging_small_penalty(self):
        from core.freshness_tracker import FreshnessTracker
        tracker = FreshnessTracker()
        aging = datetime.now(timezone.utc) - timedelta(hours=15)
        result = tracker.evaluate("PubMed", aging)
        assert result.freshness_label == "AGING"
        assert result.staleness_penalty == 0.02


# ── Fix #9: Audit Trail ──────────────────────────────────────────────────────

class TestAuditTrail:
    def test_audit_entry_has_signature(self):
        from core.audit_trail import AuditEntry
        entry = AuditEntry(
            run_id="test-run-12345678",
            timestamp="2024-01-01T00:00:00Z",
            query="test query",
            intent="drug_interaction",
            drug_names=["warfarin"],
            sources_queried=["FDA"],
            docs_retrieved=3,
            docs_used=2,
            fallback_triggered=False,
            retrieval_rounds=1,
            rewrite_strategies_used=[],
            relevance_scores=[],
            conflict_detected=False,
            conflicts=[],
            staleness_flags=[],
            srag_iterations=1,
            hallucination_flags=[],
            hallucination_repaired=False,
            quality_score=0.8,
            confidence_score=0.75,
            claim_evidence=[],
            response_preview="Test response",
            risk_level="LOW",
            processing_time_ms=1500,
        )
        assert entry.audit_signature.startswith("ARAG-")

    def test_evidence_chain_builds(self):
        from core.audit_trail import AuditTrail
        trail = AuditTrail()
        docs = [{"source": "FDA", "content": "Warfarin is an anticoagulant used to prevent blood clots.", "url": "https://fda.gov"}]
        response = "Warfarin is an anticoagulant used to prevent blood clots in patients."
        evidence = trail.build_evidence_chain(response, docs)
        assert isinstance(evidence, list)


# ── Fix #10: Quality Gate ────────────────────────────────────────────────────

class TestQualityGate:
    def test_quality_labels_coverage(self):
        from agents.quality_gate import QUALITY_LABELS
        assert len(QUALITY_LABELS) == 5
        labels = set(QUALITY_LABELS.values())
        assert "EXCELLENT" in labels
        assert "REJECTED" in labels

    def test_default_pass(self):
        from agents.quality_gate import QualityGate
        gate = QualityGate()
        result = gate._default_pass("test response")
        assert result.passed_gate == True
        assert result.overall_score > 0

    def test_refusal_response_exists(self):
        from agents.quality_gate import REFUSAL_RESPONSE
        assert "pharmacist" in REFUSAL_RESPONSE.lower()
        assert "PubMed" in REFUSAL_RESPONSE


# ── Integration Tests (require GROQ_API_KEY) ───────────────────────────────

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY"
)
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_drug_interaction(self):
        from core.arag_pipeline import ARAGPipeline
        pipeline = ARAGPipeline()
        result = await pipeline.run("What are drug interactions between warfarin and aspirin?")
        assert result.answer
        assert result.intent == "drug_interaction"
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.quality_score <= 1.0
        assert result.audit_id.startswith("ARAG-")

    @pytest.mark.asyncio
    async def test_full_pipeline_clinical_trial(self):
        from core.arag_pipeline import ARAGPipeline
        pipeline = ARAGPipeline()
        result = await pipeline.run("Active clinical trials for lung cancer")
        assert result.intent == "clinical_trial"
        assert result.answer
        assert result.quality_label in ["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR", "REJECTED"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
