from spark_doctor.models import Finding, ScanReport
from spark_doctor.rules.engine import Rule, run_rules


def _rule(rule_id: str, severity: str) -> Rule:
    return Rule(rule_id, rule_id, lambda _: [Finding(rule_id=rule_id, title=rule_id, severity=severity)])


def test_findings_are_ordered_most_severe_first() -> None:
    rules = [_rule("a", "info"), _rule("b", "warning"), _rule("c", "critical"), _rule("d", "warning")]
    findings = run_rules(ScanReport(), rules)
    assert [f.rule_id for f in findings] == ["c", "b", "d", "a"]
