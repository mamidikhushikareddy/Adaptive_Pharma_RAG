"""ARAG_PHARMA — CLI Demo"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from core.arag_pipeline import ARAGPipeline

console = Console()

DEMOS = [
    "What are the drug interactions between warfarin and aspirin?",
    "Find active recruiting clinical trials for non-small cell lung cancer with pembrolizumab",
    "What adverse events have been reported for metformin in elderly patients with renal impairment?",
]


def print_result(query: str, r):
    console.print(Panel(r.answer[:600] + ("..." if len(r.answer) > 600 else ""),
        title=f"[bold green]Answer[/] | Confidence: {r.confidence_score:.0%} | Quality: {r.quality_label}",
        border_style="green"))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="bold cyan", width=28)
    table.add_column("Value")

    table.add_row("Audit ID", f"[yellow]{r.audit_id}[/]")
    table.add_row("Intent (Fix #7)", r.intent.replace("_", " ").title())
    table.add_row("Drugs Detected", ", ".join(r.drug_names) or "None")
    table.add_row("High-Risk Flag", "⚠️ YES" if r.is_high_risk else "✅ No")
    table.add_row("─"*28, "─"*30)
    table.add_row("Fix #1 Triple-layer", f"{len(r.triple_layer_scores)} docs evaluated")
    table.add_row("Fix #2 Confidence", f"{r.confidence_score:.0%}")
    table.add_row("Fix #3 Retrieval rounds", str(r.retrieval_rounds))
    table.add_row("Fix #3 Rewrites used", ", ".join(r.rewrite_strategies_used) or "None")
    table.add_row("Fix #4 Sources approved", str(len(r.sources)))
    table.add_row("Fix #5 Hallucination score", f"{r.hallucination_score:.0%}")
    table.add_row("Fix #5 Auto-repaired", "Yes" if r.hallucination_repaired else "No")
    table.add_row("Fix #5 Hard caveat", "Yes" if r.hard_caveat_added else "No")
    table.add_row("Fix #6 Conflicts detected", str(len(r.conflicts)))
    table.add_row("Fix #8 Freshness", str(r.freshness_summary))
    table.add_row("Fix #9 Evidence chain", f"{len(r.evidence_chain)} claims mapped")
    table.add_row("Fix #10 Quality gate", f"{r.quality_label} ({r.quality_score:.0%})")
    table.add_row("─"*28, "─"*30)
    table.add_row("SRAG iterations", str(r.srag_iterations))
    table.add_row("Context supported", "✅ Yes" if r.is_supported else "❌ No")
    table.add_row("Risk level", r.risk_level)
    table.add_row("Processing time", f"{r.processing_time_ms}ms")
    table.add_row("Professional consult", "⚠️ Required" if r.requires_professional else "Optional")

    console.print(table)
    console.print(f"\n[dim]{r.disclaimer}[/dim]")


async def run_demo(query: str = None):
    console.print(Panel.fit(
        "[bold cyan]🧬 ARAG_PHARMA v3[/]\n"
        "[dim]All 10 Fixes Active — 100% Free Stack (Groq)[/]\n"
        "[yellow]#1 Triple-Eval  #2 Continuous Conf  #3 Anti-Loop  #4 Approved Sources[/]\n"
        "[yellow]#5 Halluc Guard  #6 Conflict Det  #7 MedDRA  #8 Freshness  #9 Audit  #10 Quality[/]",
        border_style="cyan"
    ))

    pipeline = ARAGPipeline()
    queries = [query] if query else DEMOS

    for i, q in enumerate(queries, 1):
        console.rule(f"Demo {i}/{len(queries)}")
        console.print(f"[bold yellow]Query:[/] {q}")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as p:
            task = p.add_task("Running ARAG_PHARMA v3 pipeline...", total=None)
            try:
                result = await pipeline.run(q)
                p.stop()
                print_result(q, result)
            except Exception as e:
                p.stop()
                console.print(f"[red]Error: {e}[/]")

        if i < len(queries):
            console.print("\n[dim]Press Enter for next...[/dim]")
            input()

    console.print("[bold green]✅ All demos complete![/]")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    asyncio.run(run_demo(q))
