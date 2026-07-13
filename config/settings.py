"""
ARAG_PHARMA — Central Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / "config" / ".env")


class Settings:
    # ── Groq LLM ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_STRONG_MODEL: str = os.getenv("LLM_STRONG_MODEL", "llama-3.3-70b-versatile")
    LLM_FAST_MODEL: str = os.getenv("LLM_FAST_MODEL", "llama-3.1-8b-instant")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MAX_RETRIES: int = 3
    GROQ_RETRY_DELAY: float = 2.0
    GROQ_REQUEST_TIMEOUT: float = 30.0

    # ── Embeddings (local) ────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # ── Pharma APIs (free, no key) ────────────────────────────────────────────
    FDA_API_BASE: str = "https://api.fda.gov"
    PUBMED_API_BASE: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    CLINICALTRIALS_API: str = "https://clinicaltrials.gov/api/v2"
    PUBMED_EMAIL: str = os.getenv("PUBMED_EMAIL", "arag_pharma@example.com")

    # ── Web Fallback ──────────────────────────────────────────────────────────
    ENABLE_WEB_FALLBACK: bool = os.getenv("ENABLE_WEB_FALLBACK", "true").lower() == "true"
    # Trigger web fallback when RELEVANT (not just total) docs < this threshold
    WEB_FALLBACK_MIN_DOCS: int = int(os.getenv("WEB_FALLBACK_MIN_DOCS", "2"))
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    WEB_SOURCE_TAG: str = "WebSearch"

    # ── Confidence Thresholds ─────────────────────────────────────────────────
    CONFIDENCE_GENERATE_THRESHOLD: float = 0.50   # FIX: Lowered from 0.55 — web docs score lower but are valid
    CONFIDENCE_FETCH_MORE_THRESHOLD: float = 0.30
    CONFIDENCE_REWRITE_THRESHOLD: float = 0.25    # FIX: Lowered — prevents excessive rewriting for valid queries

    # ── Anti-loop ─────────────────────────────────────────────────────────────
    MAX_RETRIEVAL_ROUNDS: int = int(os.getenv("MAX_RETRIEVAL_ROUNDS", "3"))   # FIX: Reduced from 4, saves time
    MAX_REWRITE_STRATEGIES: int = 4
    HARD_STOP_ITERATIONS: int = 3   # FIX: Reduced from 4, stops earlier when web fallback has fired

    # ── Approved Sources ──────────────────────────────────────────────────────
    APPROVED_SOURCES = ["FDA", "PubMed", "ClinicalTrials.gov", "FAERS", "DailyMed"]

    # ── Hallucination ─────────────────────────────────────────────────────────
    ENABLE_HALLUCINATION_CHECK: bool = True
    HALLUCINATION_AUTO_REPAIR: bool = True
    HALLUCINATION_HARD_CAVEAT_THRESHOLD: float = 0.75

    # ── Conflict Detection ────────────────────────────────────────────────────
    ENABLE_CONFLICT_DETECTION: bool = True

    # ── Freshness ─────────────────────────────────────────────────────────────
    DATA_STALENESS_HOURS: int = 48
    STALENESS_CONFIDENCE_PENALTY: float = 0.10

    # ── Audit Trail ───────────────────────────────────────────────────────────
    ENABLE_AUDIT_TRAIL: bool = True
    AUDIT_LOG_PATH: str = str(BASE_DIR / "logs" / "audit.jsonl")

    # ── Quality Gate ──────────────────────────────────────────────────────────
    ENABLE_QUALITY_GATE: bool = True
    QUALITY_GATE_MIN_SCORE: float = 0.35   # FIX: Lowered from 0.40 — knowledge-based answers score lower

    # ── Triple-Layer Eval ─────────────────────────────────────────────────────
    ENABLE_TRIPLE_EVAL: bool = True

    # ── SRAG ──────────────────────────────────────────────────────────────────
    ENABLE_SELF_REFLECTION: bool = True
    SRAG_MAX_ITERATIONS: int = 2

    # ── Server ────────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Intent Categories ─────────────────────────────────────────────────────
    INTENT_CATEGORIES = [
        "drug_interaction", "ingredient_interaction", "clinical_trial", "adverse_event",
        "drug_label", "literature_search", "dosage_information",
        "pharmacokinetics", "regulatory_compliance", "general_pharma",
    ]

    # ── Source Trust Scores ───────────────────────────────────────────────────
    SOURCE_TRUST_SCORES = {
        "FDA": 0.97, "FAERS": 0.93, "DailyMed": 0.95,
        "PubMed": 0.88, "ClinicalTrials.gov": 0.90,
        "WebSearch": 0.62,   # Internet sources — valid but less authoritative
    }

    # ── High-risk Drugs ───────────────────────────────────────────────────────
    HIGH_RISK_DRUGS = [
        "warfarin", "heparin", "methotrexate", "lithium", "digoxin",
        "insulin", "phenytoin", "theophylline", "aminoglycoside",
        "chemotherapy", "immunosuppressant", "clozapine", "tacrolimus",
        "cyclosporine", "vancomycin", "amiodarone",
    ]

    SYSTEM_PROMPT = """You are ARAG_PHARMA v3, an advanced pharmaceutical and cosmetic ingredient intelligence system.
You provide accurate, evidence-based information about pharmaceutical drugs AND cosmetic/OTC ingredients.
Your knowledge covers: prescription drugs, OTC medications, cosmetic actives (niacinamide, retinol, AHAs, BHAs,
hyaluronic acid, salicylic acid, etc.), nutraceuticals, and supplements.

Rules:
1. Always provide substantive, specific answers — never refuse to answer if you have relevant knowledge
2. Clearly label every key claim with its source: [FDA Drug Label], [PubMed], [Knowledge Base], [Web: provider]
3. Never fabricate specific clinical trial IDs, PMID numbers, or dosages not in context
4. Recommend professional consultation but ALWAYS provide the actual information first
5. For cosmetic ingredient combinations: discuss individual effects + synergies + cautions + application order"""

    GROQ_AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]


settings = Settings()
