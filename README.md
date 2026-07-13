# 🧬 ARAG_PHARMA — Adaptable Retreival Augmented generation for Pharmaceuticals

> **Powered by: Groq (free) · FDA · PubMed · ClinicalTrials.gov · Local Embeddings**  
> 
---

## 🏗️ Architecture

```
USER QUERY
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 1: QUERY ANALYZER [Groq: llama-3.1-8b-instant]    │
│  • 9-intent classification                               │
│  • Drug name extraction                                  │
│  • MedDRA terminology injection (Fix #7)                 │
│  • High-risk drug detection                              │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 2: MULTI-SOURCE LOADER [All FREE APIs]             │
│  ✅ FDA OpenFDA (drug labels + FAERS)                   │
│  ✅ PubMed NCBI (35M+ articles)                         │
│  ✅ ClinicalTrials.gov (400K+ trials)                   │
│  ✅ Generic web search                                  │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 3: TRIPLE-LAYER EVAL [Groq: fast model]            │
│  Layer 1 [50%]: Semantic LLM relevance                   │
│  Layer 2 [25%]: Rule-based source trust (FDA=97%)        │
│  Layer 3 [25%]: Cross-doc consistency LLM                │
│                                                          │
│  Routing: >55%→GENERATE | 30-55%→FETCH | <30%→REWRITE    │ 
│  Anti-loop: query registry + 4 strategies + hard stop    │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  STEPS 4-5: FRESHNESS + CONFLICT [Groq + rules]          │
│  Fix #8: FRESH/AGING/STALE/VERY_STALE penalties          │
│  Fix #6: LLM conflict detection + trust resolution       │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 6: SRAG GENERATION [Groq: llama-3.3-70b]           │
│  [Retrieve] → [IsSUP] → [IsUSE] → max 2 iterations       │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 7: HALLUCINATION CHECK [Groq + regex]              │
│  Regex scan (dosages, PMIDs, NCTs) + LLM cross-check     │
│  Auto-repair + hard caveat injection                     │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 8: QUALITY GATE [Groq] + AUDIT TRAIL [local]       │
│  6-dimension 0-100% score + refusal if too low           │
│  Full evidence chain logged to audit.jsonl               │
└────────────────────────────┬─────────────────────────────┘
                             ▼
                    ARAG_RESPONSE
```
---

## 🚀 Quick Start (5 minutes)

### Step 1: Get Groq API Key (FREE)
1. Go to **https://console.groq.com**
2. Sign up (no credit card needed)
3. Click **"API Keys"** → **"Create API Key"**
4. Copy your key (starts with `gsk_`)

### Step 2: Install
```bash
unzip ARAG_PHARMA_V3.zip
cd ARAG_PHARMA_V3
**'''Environment setup'''**
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Note**: First run downloads the embedding model (~80MB) automatically. Needs internet once.

### Step 3: Configure
```bash
cp config/.env.example config/.env
# Edit config/.env and paste your GROQ_API_KEY
```

### Step 4: Run

**Option A — Streamlit UI (recommended):**
```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

**Option B — FastAPI Server:**
```bash
python api/server.py
# API: http://localhost:8000
# Swagger: http://localhost:8000/docs
# Recent audit: http://localhost:8000/audit
```

**Option C — CLI Demo:**
```bash
python scripts/demo.py
# Or with custom query:
python scripts/demo.py "What are the interactions between warfarin and aspirin?"
```

**Option D — Run Tests:**
```bash
# Unit tests (no API key needed):
pytest tests/test_all.py -v -k "not Integration"

# Integration tests (needs GROQ_API_KEY in env):
pytest tests/test_all.py -v
```

---

## 🤖 Free Groq Models Available

| Model | Quality | Speed | Context |
|---|---|---|---|
| `llama-3.3-70b-versatile` | ⭐⭐⭐⭐⭐ Best | Medium | 8K |
| `llama3-70b-8192` | ⭐⭐⭐⭐ Great | Medium | 8K |
| `mixtral-8x7b-32768` | ⭐⭐⭐⭐ Great | Medium | **32K** |
| `gemma2-9b-it` | ⭐⭐⭐ Good | Fast | 8K |
| `llama-3.1-8b-instant` | ⭐⭐⭐ Good | **Fastest** | 8K |

Set your preferred models in `config/.env`:
```env
LLM_MODEL=llama-3.3-70b-versatile      # Main model
LLM_STRONG_MODEL=llama-3.3-70b-versatile  # Safety checks
LLM_FAST_MODEL=llama-3.1-8b-instant    # Quick tasks
```

---

## 📁 Project Structure

```
ARAG_PHARMA_V3/
├── core/
│   ├── groq_client.py          ← Groq wrapper (rate-limit retry, JSON parse, fallback)
│   ├── arag_pipeline.py        ← Master pipeline (all 10 fixes)
│   ├── srag_module.py          ← Self-RAG + Groq generation
│   ├── freshness_tracker.py    ← Fix #8: Staleness tracking
│   └── audit_trail.py          ← Fix #9: Evidence chain + logging
├── agents/
│   ├── query_analyzer.py       ← Fix #7: 9-intent + MedDRA (Groq)
│   ├── triple_layer_evaluator.py ← Fix #1+2: Triple eval (Groq + rules)
│   ├── anti_loop_rewriter.py   ← Fix #3: Anti-loop + 4 strategies (Groq)
│   ├── conflict_detector.py    ← Fix #6: Source conflicts (Groq)
│   ├── hallucination_checker.py ← Fix #5: Hallucination guard (Groq + regex)
│   └── quality_gate.py         ← Fix #10: Quality assessment (Groq)
├── data/
│   ├── pharma_loader.py        ← Fix #4: Approved sources orchestrator
│   ├── fda_client.py           ← FDA OpenFDA + FAERS (free)
│   ├── pubmed_client.py        ← PubMed NCBI (free)
│   └── clinicaltrials_client.py ← ClinicalTrials.gov (free)
├── api/server.py               ← FastAPI REST server
├── ui/app.py                   ← Streamlit dashboard
├── config/
│   ├── settings.py             ← All configuration
│   └── .env.example            ← Template (copy to .env)
├── tests/test_all.py           ← Tests for all 10 fixes
├── scripts/demo.py             ← Rich CLI demo
├── logs/audit.jsonl            ← Auto-created audit log
└── requirements.txt
```

---

## 🔑 Sample Queries

```
1. "What are the drug interactions between warfarin and aspirin?"
2. "Find recruiting clinical trials for non-small cell lung cancer with pembrolizumab"
3. "What adverse events have been reported for metformin in elderly patients?"
4. "FDA-approved dosage for methotrexate in rheumatoid arthritis"
5. "Pharmacokinetics of vancomycin in renal impairment"
6. "GLP-1 receptor agonists cardiovascular outcomes meta-analysis"
7. "FDA regulatory approval pathway for biosimilars"
```

---

## ⚠️ Disclaimer

For educational and research purposes only. Not medical advice.
Always consult a licensed healthcare professional for clinical decisions.
