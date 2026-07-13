"""ARAG_PHARMA— FastAPI Server"""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from core.arag_pipeline import ARAGPipeline, ARAGResponse
from core.audit_trail import audit_trail
from config.settings import settings

app = FastAPI(
    title="ARAG_PHARMA v3",
    description="Advanced Pharma RAG — Groq + 100% Free Stack + All 10 Loophole Fixes",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline: ARAGPipeline | None = None


@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = ARAGPipeline()
    logger.info("ARAG_PHARMA v3 API ready")


class QueryRequest(BaseModel):
    query: str
    verbose: bool = False


@app.get("/")
async def root():
    return {
        "name": "ARAG_PHARMA v3",
        "version": "3.0.0",
        "fixes_active": [
            "Fix #1: Triple-layer evaluation",
            "Fix #2: Continuous confidence + routing",
            "Fix #3: Anti-loop + hard stop",
            "Fix #4: Approved sources only",
            "Fix #5: Post-gen hallucination check",
            "Fix #6: Conflict detection",
            "Fix #7: 9-intent + MedDRA terminology",
            "Fix #8: Data freshness tracking",
            "Fix #9: Full audit trail",
            "Fix #10: Quality gate",
        ],
        "endpoints": {
            "POST /query": "Submit pharmaceutical query",
            "GET /health": "Health check",
            "GET /audit": "Recent audit log",
            "GET /demo": "Run demo query",
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "pipeline_ready": pipeline is not None}


@app.post("/query")
async def query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if len(request.query) > 1000:
        raise HTTPException(400, "Query too long (max 1000 chars)")
    try:
        result: ARAGResponse = await pipeline.run(request.query)
        response_dict = {
            "answer": result.answer,
            "disclaimer": result.disclaimer,
            "is_refused": result.is_refused,
            "audit_id": result.audit_id,
            "intent": result.intent,
            "drug_names": result.drug_names,
            "is_high_risk": result.is_high_risk,
            "confidence_score": result.confidence_score,
            "quality_score": result.quality_score,
            "quality_label": result.quality_label,
            "quality_dimensions": result.quality_dimensions,
            "hallucination_score": result.hallucination_score,
            "hallucination_repaired": result.hallucination_repaired,
            "hard_caveat_added": result.hard_caveat_added,
            "conflicts": result.conflicts,
            "freshness_summary": result.freshness_summary,
            "staleness_warnings": result.staleness_warnings,
            "sources": result.sources,
            "requires_professional": result.requires_professional,
            "triple_layer_scores": result.triple_layer_scores,
            "retrieval_rounds": result.retrieval_rounds,
            "rewrite_strategies_used": result.rewrite_strategies_used,
            "srag_iterations": result.srag_iterations,
            "is_supported": result.is_supported,
            "risk_level": result.risk_level,
            "processing_time_ms": result.processing_time_ms,
            "evidence_chain": result.evidence_chain[:5],
        }
        if request.verbose:
            response_dict["critique_log"] = result.critique_log
        return response_dict
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(500, f"Pipeline error: {str(e)}")


@app.get("/audit")
async def get_audit(n: int = 10):
    """Get recent audit entries (Fix #9)."""
    return {"recent_runs": audit_trail.get_recent_runs(n)}


@app.get("/demo")
async def demo():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    result = await pipeline.run("What are the drug interactions between warfarin and aspirin?")
    return {
        "demo_query": "What are the drug interactions between warfarin and aspirin?",
        "answer_preview": result.answer[:300] + "...",
        "confidence": f"{result.confidence_score:.0%}",
        "quality": f"{result.quality_label} ({result.quality_score:.0%})",
        "audit_id": result.audit_id,
        "fixes_active": result.retrieval_rounds,
        "time_ms": result.processing_time_ms,
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
