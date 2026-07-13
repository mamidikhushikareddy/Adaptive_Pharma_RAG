"""
ARAG_PHARMA — Streamlit UI

"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from core.arag_pipeline import ARAGPipeline, ARAGResponse
from core.audit_trail import audit_trail
from config.settings import settings

st.set_page_config(
    page_title="ARAG_PHARMA v3",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
  background: linear-gradient(135deg, #0a0a1a, #1a0a2e, #0a1a3e);
  padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1rem;
}
.fix-badge {
  display: inline-block; background: #1e3a5f; color: #90caf9;
  border-radius: 20px; padding: 3px 10px; font-size: 0.75rem; margin: 2px;
}
.web-badge {
  display: inline-block; background: #3d2a00; color: #ffcc80;
  border-radius: 20px; padding: 3px 10px; font-size: 0.75rem; margin: 2px;
}
.quality-EXCELLENT { color: #00c853; font-weight: bold; }
.quality-GOOD { color: #69f0ae; font-weight: bold; }
.quality-ACCEPTABLE { color: #ffd740; font-weight: bold; }
.quality-POOR { color: #ff6d00; font-weight: bold; }
.quality-REJECTED { color: #d50000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">
  <h1 style="color:white;margin:0">🧬 ARAG_PHARMA <span style="color:#64b5f6">v3</span></h1>
  <p style="color:#90caf9;margin:4px 0">All 10 CRAG Loophole Fixes + Internet Search Fallback</p>
  <div>
    <span class="fix-badge">#1 Triple-Layer Eval</span>
    <span class="fix-badge">#2 Continuous Confidence</span>
    <span class="fix-badge">#3 Anti-Loop</span>
    <span class="fix-badge">#4 Approved Sources</span>
    <span class="fix-badge">#5 Hallucination Guard</span>
    <span class="fix-badge">#6 Conflict Detection</span>
    <span class="fix-badge">#7 9-Intent + MedDRA</span>
    <span class="fix-badge">#8 Freshness Tracking</span>
    <span class="fix-badge">#9 Audit Trail</span>
    <span class="fix-badge">#10 Quality Gate</span>
    <span class="web-badge">🌐 Web Fallback</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── API Key Check ─────────────────────────────────────────────────────────────
if not settings.GROQ_API_KEY:
    st.error(
        "⚠️ **GROQ_API_KEY is not set!**\n\n"
        "1. Get a free key at https://console.groq.com\n"
        "2. Copy `config/.env.example` to `config/.env`\n"
        "3. Set `GROQ_API_KEY=gsk_your_key_here` in `config/.env`\n"
        "4. Restart the app"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    verbose = st.toggle("Verbose (SRAG critique log)", False)

    st.divider()
    st.header("📋 Sample Queries")
    samples = {
        "💊 Drug Interaction": "What are the drug interactions between warfarin and aspirin?",
        "🔬 Clinical Trial": "Find recruiting trials for non-small cell lung cancer with pembrolizumab",
        "⚠️ Adverse Events": "What adverse events have been reported for metformin in elderly patients?",
        "📋 Dosage": "FDA-approved dosage for methotrexate in rheumatoid arthritis",
        "📚 Literature": "GLP-1 receptor agonists cardiovascular outcomes meta-analysis",
        "⚗️ PK/PD": "Pharmacokinetics of vancomycin in renal impairment",
        "📜 Regulatory": "What is the FDA approval status of pembrolizumab for melanoma?",
    }
    for label, sample in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state.sq = sample

    st.divider()
    st.header("📖 Data Sources")
    st.markdown(
        "**Approved (High Trust):**\n"
        "✅ FDA OpenFDA\n✅ FDA FAERS\n✅ PubMed (35M+)\n✅ ClinicalTrials.gov\n\n"
        "**Fallback (Lower Trust):**\n"
        f"{'🌐 Internet Search (active)' if settings.ENABLE_WEB_FALLBACK else '❌ Internet Search (disabled)'}"
    )

    st.divider()
    st.header("🔍 Recent Audit Runs")
    for run in audit_trail.get_recent_runs(3):
        st.caption(
            f"`{run.get('audit_signature', '?')}` | {run.get('intent', '?')} | "
            f"{run.get('confidence_score', 0):.0%} conf"
        )


@st.cache_resource
def get_pipeline():
    return ARAGPipeline()


# ── Query Input ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_area(
        "🔍 Pharmaceutical Query:",
        value=st.session_state.get("sq", ""),
        height=90,
        placeholder="e.g., What are the drug interactions between warfarin and aspirin?",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run ARAG", type="primary", use_container_width=True)
    st.caption("FDA + PubMed + ClinicalTrials")

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_btn and query.strip():
    pipeline = get_pipeline()
    progress = st.empty()
    steps = [
        "🔍 [1/8] Analyzing query + intent classification...",
        "📥 [2/8] Loading approved pharma sources (+ web fallback if needed)...",
        "⚖️ [3/8] Running triple-layer evaluation...",
        "⏰ [4/8] Checking data freshness...",
        "⚔️ [5/8] Detecting source conflicts...",
        "🤖 [6/8] SRAG generation + self-reflection...",
        "🛡️ [7/8] Post-generation hallucination check...",
        "🏆 [8/8] Quality gate assessment...",
    ]
    for step in steps:
        progress.info(step)

    try:
        result: ARAGResponse = asyncio.run(pipeline.run(query))
        progress.empty()
    except RuntimeError as e:
        progress.empty()
        if "GROQ_API_KEY" in str(e):
            st.error(f"🔑 API Key Error: {e}")
        else:
            st.error(f"❌ Pipeline error: {e}")
        st.stop()
    except Exception as e:
        progress.empty()
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

    st.divider()

    # ── Summary Metrics Row ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Confidence", f"{result.confidence_score:.0%}")
    m2.metric("Quality", f"{result.quality_score:.0%} ({result.quality_label})")
    m3.metric("Hallucination", f"{result.hallucination_score:.0%}")
    m4.metric("CRAG Rounds", result.retrieval_rounds)
    m5.metric("SRAG Iters", result.srag_iterations)
    m6.metric("Conflicts", len(result.conflicts))
    m7.metric("Time", f"{result.processing_time_ms}ms")

    # Web fallback banner
    if result.used_web_fallback:
        st.warning(
            "🌐 **Internet Search Fallback Was Used** — Approved pharma databases returned insufficient "
            "results. Some information was sourced from general internet search (lower trust). "
            "Claims labelled **[Web]** should be independently verified."
        )

    st.markdown("---")

    # ── FIX: TWO SEPARATE TABS — Answer vs RAG Performance ───────────────────
    tab_answer, tab_performance = st.tabs(["📋 Answer", "📊 RAG Performance"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: ANSWER
    # ═════════════════════════════════════════════════════════════════════════
    with tab_answer:
        # Status badges
        badge_cols = st.columns(4)
        badge_cols[0].markdown(f"**Quality:** `{result.quality_label}`")
        badge_cols[1].markdown(f"**Risk:** `{result.risk_level}`")
        badge_cols[2].markdown(f"**Intent:** `{result.intent.replace('_',' ').title()}`")
        badge_cols[3].markdown(f"**Audit ID:** `{result.audit_id}`")

        st.markdown("---")

        if result.is_refused:
            st.error("🚫 **Response Refused by Quality Gate** — insufficient evidence quality")

        # Main answer
        st.subheader("🧬 Response")
        st.markdown(result.answer)

        # Disclaimer
        st.info(result.disclaimer)

        st.markdown("---")

        # Sources — clearly split pharma vs web
        col_src1, col_src2 = st.columns(2)
        with col_src1:
            with st.expander(f"📚 Pharma Database Sources ({len(result.sources)})", expanded=True):
                if result.sources:
                    for src in result.sources:
                        parts = src.split(": ", 1)
                        if len(parts) == 2 and parts[1].startswith("http"):
                            st.markdown(f"✅ **{parts[0]}** — [{parts[1][:50]}...]({parts[1]})")
                        else:
                            st.markdown(f"✅ {src}")
                else:
                    st.caption("No authoritative pharma database sources used.")

        with col_src2:
            with st.expander(
                f"🌐 Internet Sources ({len(result.web_sources)})",
                expanded=bool(result.web_sources),
            ):
                if result.web_sources:
                    st.warning("⚠️ Internet sources have lower trust. Verify claims independently.")
                    for src in result.web_sources:
                        parts = src.split(": ", 1)
                        if len(parts) == 2 and parts[1].startswith("http"):
                            st.markdown(f"🌐 **{parts[0]}** — [{parts[1][:50]}...]({parts[1]})")
                        else:
                            st.markdown(f"🌐 {src}")
                else:
                    st.success("✅ No internet sources used — all data from authoritative pharma databases.")

        # Drug names
        if result.drug_names:
            st.subheader("💊 Drugs Identified")
            drug_html = " ".join(
                f'<span style="background:{"#f8d7da" if result.is_high_risk else "#d4edda"};'
                f'padding:4px 12px;border-radius:20px;margin:2px">{drug}</span>'
                for drug in result.drug_names
            )
            st.markdown(drug_html, unsafe_allow_html=True)
            if result.is_high_risk:
                st.warning("⚠️ High-risk drug(s) identified — extra clinical caution required.")

        # Conflicts
        if result.conflicts:
            with st.expander(f"⚔️ Source Conflicts Detected ({len(result.conflicts)})", expanded=True):
                for c in result.conflicts:
                    severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "CRITICAL": "⛔"}.get(c["severity"], "⚪")
                    st.warning(f"{severity_icon} **{c['topic']}** [{c['severity']}]\n\n{c['resolution']}")

        # Staleness warnings
        if result.staleness_warnings:
            with st.expander(f"⏰ Data Freshness Warnings ({len(result.staleness_warnings)})"):
                for w in result.staleness_warnings:
                    st.warning(w)

        # Hallucination caveat
        if result.hallucination_repaired or result.hard_caveat_added:
            with st.expander("🛡️ Hallucination Guard Report"):
                st.metric("Hallucination Score", f"{result.hallucination_score:.0%}")
                if result.hallucination_repaired:
                    st.success("✅ Auto-repair applied to response")
                if result.hard_caveat_added:
                    st.warning("⚠️ Hard accuracy caveat was added to response")

        # Verbose SRAG log
        if verbose and result.critique_log:
            with st.expander("🧠 SRAG Critique Log"):
                for entry in result.critique_log:
                    st.json(entry)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: RAG PERFORMANCE (all metrics, fully collapsable sections)
    # ═════════════════════════════════════════════════════════════════════════
    with tab_performance:
        st.subheader("📊 RAG Pipeline Performance Breakdown")
        st.caption(
            f"Audit: `{result.audit_id}` | "
            f"Processing time: {result.processing_time_ms}ms | "
            f"CRAG rounds: {result.retrieval_rounds} | "
            f"SRAG iterations: {result.srag_iterations}"
        )

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            # Fix #10: Quality Gate Breakdown
            with st.expander("🏆 Fix #10 — Quality Gate Breakdown", expanded=True):
                st.progress(result.quality_score, text=f"Overall: {result.quality_score:.0%} — {result.quality_label}")
                for dim, score in result.quality_dimensions.items():
                    st.progress(score, text=f"{dim.replace('_', ' ').title()}: {score:.0%}")

            # Fix #1: Triple-Layer Scores
            with st.expander("⚖️ Fix #1 — Triple-Layer Evaluation Scores", expanded=True):
                if result.triple_layer_scores:
                    for ts in result.triple_layer_scores[:8]:
                        status = "❌ Discarded" if ts.get("discarded") else "✅ Used"
                        st.markdown(
                            f"**{ts['source']}** {status}\n\n"
                            f"Final: `{ts['final_score']:.2f}` | "
                            f"Semantic: `{ts['semantic']:.2f}` | "
                            f"Trust: `{ts['trust']:.2f}` | "
                            f"Consistency: `{ts['consistency']:.2f}`"
                        )
                        st.divider()
                else:
                    st.info("No triple-layer evaluation data available.")

            # Fix #5: Hallucination Details
            with st.expander("🛡️ Fix #5 — Hallucination Analysis"):
                st.metric("Hallucination Score", f"{result.hallucination_score:.0%}")
                st.metric("Auto-Repaired", "Yes" if result.hallucination_repaired else "No")
                st.metric("Hard Caveat Added", "Yes" if result.hard_caveat_added else "No")
                st.progress(
                    1.0 - result.hallucination_score,
                    text=f"Grounding: {1.0 - result.hallucination_score:.0%}"
                )

        with perf_col2:
            # Fix #2 & #3: Confidence + Anti-Loop
            with st.expander("🎯 Fix #2+#3 — Confidence & Anti-Loop", expanded=True):
                st.metric("Overall Confidence", f"{result.confidence_score:.0%}")
                st.metric("Retrieval Rounds", result.retrieval_rounds)
                st.metric("SRAG Iterations", result.srag_iterations)
                st.metric("Web Fallback Used", "Yes ⚠️" if result.used_web_fallback else "No ✅")
                if result.rewrite_strategies_used:
                    st.markdown("**Query Rewrite Strategies:**")
                    for s in result.rewrite_strategies_used:
                        st.markdown(f"  • `{s}`")
                else:
                    st.success("No query rewriting needed")

            # Fix #8: Freshness
            with st.expander("⏰ Fix #8 — Data Freshness Summary"):
                fs = result.freshness_summary
                cols = st.columns(4)
                cols[0].metric("🟢 Fresh", fs.get("fresh", 0))
                cols[1].metric("🟡 Aging", fs.get("aging", 0))
                cols[2].metric("🟠 Stale", fs.get("stale", 0))
                cols[3].metric("🔴 Very Stale", fs.get("very_stale", 0))

            # Fix #9: Evidence Chain
            if result.evidence_chain:
                with st.expander(f"🔗 Fix #9 — Evidence Chain ({len(result.evidence_chain)} claims)"):
                    for ev in result.evidence_chain[:6]:
                        st.markdown(
                            f"**Claim:** {ev['claim'][:100]}...\n\n"
                            f"**Source:** {ev['source']} | Confidence: {ev['confidence']:.2f}"
                        )
                        if ev.get("url"):
                            st.markdown(f"[🔗 View source]({ev['url']})")
                        st.divider()

        # Full-width sections at bottom of performance tab
        with st.expander("🔧 Pipeline Configuration (Active Settings)"):
            config_data = {
                "LLM Model": settings.LLM_MODEL,
                "Fast Model": settings.LLM_FAST_MODEL,
                "Web Fallback": "✅ Enabled" if settings.ENABLE_WEB_FALLBACK else "❌ Disabled",
                "Web Fallback Min Docs": settings.WEB_FALLBACK_MIN_DOCS,
                "Serper API": "✅ Configured" if settings.SERPER_API_KEY else "❌ Not set (using DuckDuckGo)",
                "Max Retrieval Rounds": settings.MAX_RETRIEVAL_ROUNDS,
                "SRAG Max Iterations": settings.SRAG_MAX_ITERATIONS,
                "Hallucination Check": "✅" if settings.ENABLE_HALLUCINATION_CHECK else "❌",
                "Quality Gate Min": f"{settings.QUALITY_GATE_MIN_SCORE:.0%}",
                "Conflict Detection": "✅" if settings.ENABLE_CONFLICT_DETECTION else "❌",
                "Triple-Layer Eval": "✅" if settings.ENABLE_TRIPLE_EVAL else "❌",
                "Audit Trail": "✅" if settings.ENABLE_AUDIT_TRAIL else "❌",
            }
            col_a, col_b = st.columns(2)
            items = list(config_data.items())
            for i, (k, v) in enumerate(items):
                (col_a if i % 2 == 0 else col_b).markdown(f"**{k}:** {v}")

        with st.expander("📜 Raw Source List (All Docs Retrieved)"):
            all_sources = result.sources + result.web_sources
            if all_sources:
                for src in all_sources:
                    icon = "🌐" if src in result.web_sources else "📚"
                    st.markdown(f"{icon} {src}")
            else:
                st.info("No sources were retrieved.")

elif run_btn:
    st.warning("⚠️ Please enter a pharmaceutical query before running.")

st.markdown("---")
st.markdown(
    "<center><small>ARAG_PHARMA v3 — For research use only. Not medical advice. "
    "Always consult a healthcare professional.</small></center>",
    unsafe_allow_html=True,
)
