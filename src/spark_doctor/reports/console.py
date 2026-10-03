from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..models import ScanReport

SEVERITY_COLOR = {"info": "cyan", "warning": "yellow", "critical": "red"}
SEVERITY_LABEL = {"info": "[info]", "warning": "[warning]", "critical": "[critical]"}


def overall_status(report: ScanReport) -> str:
    sev = {f.severity for f in report.findings}
    if "critical" in sev:
        return "Critical (Incomplete)" if report.incomplete else "Critical"
    if "warning" in sev:
        return "Warning (Incomplete)" if report.incomplete else "Warning"
    if report.incomplete:
        return "Incomplete"
    if "info" in sev:
        return "Info"
    return "OK"


def render_console(report: ScanReport, console: Console | None = None) -> None:
    console = console or Console()
    status = overall_status(report)
    color = {"OK": "green", "Info": "cyan", "Warning": "yellow", "Critical": "red", "Incomplete": "yellow"}[status.split()[0]]
    console.print()
    console.print(Panel.fit(Text(f"Spark Doctor Scan — Overall: {status}", style=color)))
    if report.incomplete:
        console.print("Some checks could not finish. See collector notes and rule errors; system health is not fully assessed.")

    python_info = report.cuda_env.get("python")
    interpreter = (python_info.get("executable") if isinstance(python_info, dict) else None) or report.cuda_env.get("python_executable")
    if interpreter:
        console.print(Text(f"Selected Python environment: {interpreter}"))
        if not isinstance(python_info, dict):
            console.print("The Python probe did not complete; this environment was not assessed.")
        elif python_info.get("torch_import_ok") is False:
            console.print("PyTorch could not be imported in this environment; its CUDA compatibility was not assessed.")

    peak = (report.gpu or {}).get("peak") if isinstance(report.gpu, dict) else None
    if peak:
        util = peak.get("gpu_utilization_percent")
        pwr = peak.get("gpu_power_draw_watts")
        clk = peak.get("gpu_clock_mhz")
        tmp = peak.get("gpu_temperature_c")
        console.print(Text(
            f"GPU peak during scan: util={util}%, power={pwr} W, "
            f"clock={clk} MHz, temp={tmp} C  "
            f"(samples={len(report.gpu_samples)}, sampler={report.gpu.get('sampler')})",
            style="dim",
        ))

    if not report.findings:
        if report.incomplete:
            console.print("No findings from the checks that completed.")
        else:
            console.print("[green]No issues detected by MVP rule set.[/]")
    else:
        console.print("\n[bold]Findings:[/]")
        for i, f in enumerate(report.findings, start=1):
            c = SEVERITY_COLOR.get(f.severity, "white")
            label = SEVERITY_LABEL.get(f.severity, f"[{f.severity}]")
            heading = Text(f"\n{i}. {f.title}", style="bold")
            heading.append(f"  {label}", style=c)
            heading.append(f"  ({f.rule_id})", style="not bold")
            console.print(heading)
            if f.evidence:
                console.print("   Evidence:")
                for e in f.evidence:
                    console.print(Text(f"     - {e}"))
            if f.explanation:
                console.print(Text(f"   {f.explanation}"))
            if f.recommended_actions:
                console.print("   Next steps:")
                for a in f.recommended_actions:
                    console.print(Text(f"     - {a}"))
            if f.fix_commands:
                console.print("   Suggested commands (review before running):")
                for cmd in f.fix_commands:
                    console.print(Text(f"     {cmd}", style="bold cyan"))
            if f.escalation_actions:
                console.print("   Escalation:")
                for a in f.escalation_actions:
                    console.print(Text(f"     - {a}"))

    errs = [s for s in report.collector_statuses if not s.ok or s.errors]
    if errs:
        console.print("\n[dim]Collector notes:[/]")
        for s in errs:
            console.print(Text(f"  - {s.name}: ok={s.ok}; {', '.join(s.errors)}", style="dim"))
