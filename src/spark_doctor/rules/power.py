from __future__ import annotations

from ..models import Finding, MetricSample, ScanReport
from .engine import Rule


def _consecutive(previous: MetricSample, current: MetricSample) -> bool:
    if previous.timestamp is None or current.timestamp is None:
        return previous.timestamp is None and current.timestamp is None
    try:
        gap = (current.timestamp - previous.timestamp).total_seconds()
    except TypeError:
        return False
    # Collectors poll at one-second intervals; long gaps cannot establish persistence.
    return 0 < gap <= 5


def _evaluate(report: ScanReport) -> list[Finding]:
    current_run: list[MetricSample] = []
    longest_run: list[MetricSample] = []
    for sample in report.gpu_samples:
        if not (
            sample.gpu_utilization_percent is not None and sample.gpu_utilization_percent >= 80
            and sample.gpu_power_draw_watts is not None and sample.gpu_power_draw_watts <= 25
        ):
            current_run = []
            continue
        if current_run and not _consecutive(current_run[-1], sample):
            current_run = []
        current_run.append(sample)
        if len(current_run) > len(longest_run):
            longest_run = current_run
    if not longest_run:
        return []

    low_clock = all(
        (s.gpu_clock_mhz is not None and s.gpu_clock_mhz <= 800) for s in longest_run
    )

    sustained = len(longest_run) >= 3
    severity = "critical" if sustained else "warning"
    confidence = "high" if (sustained and low_clock) else ("medium" if sustained else "low")

    evidence = [f"Longest qualifying run: {len(longest_run)} consecutive samples."]
    for s in longest_run[:5]:
        util = s.gpu_utilization_percent
        power = s.gpu_power_draw_watts
        clock = s.gpu_clock_mhz
        evidence.append(
            f"GPU util {util:.0f}%, power {power:.1f} W, clock {clock if clock is not None else 'n/a'} MHz"
        )

    return [
        Finding(
            rule_id="power.low_draw_under_load",
            title="Possible GPU low-power state",
            severity=severity,
            confidence=confidence,
            evidence=evidence,
            explanation=(
                "Your GPU reports high utilization and unusually low power across consecutive "
                "samples. This may indicate a low-power state, but does not establish the cause."
                if sustained else
                "Your GPU reports high utilization and unusually low power in brief or separated "
                "samples. These readings do not establish a sustained low-power state or its cause."
            ),
            recommended_actions=[
                "Confirm the readings repeat during a sustained GPU workload before attempting recovery.",
                "Save all running work.",
                "Shut down the Spark.",
                "Unplug the power brick from the wall and from the Spark.",
                "Wait 60 seconds.",
                "Plug in and boot.",
                "Run `spark-doctor scan` again.",
                "Check DGX Dashboard for updates.",
            ] if sustained else [
                "Confirm a sustained GPU workload is running; short or intermittent load can be misleading.",
                "Collect more samples with `spark-doctor scan --sample-seconds 30` while the workload runs.",
                "Compare power and clock readings across the samples before deciding on recovery steps.",
            ],
            escalation_actions=[
                "If this repeats after updates, run NVIDIA Field Diagnostics and include this Spark Doctor report in a forum or support post.",
            ] if sustained else [],
            source_note="DGX Spark forum reports of 14 W power cap and low-power states under load.",
        )
    ]


rule_power_low_draw_under_load = Rule(
    id="power.low_draw_under_load",
    title="Possible GPU low-power state",
    fn=_evaluate,
)
