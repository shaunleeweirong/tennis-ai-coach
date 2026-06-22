from types import SimpleNamespace

from tennis_cv.coaching import (
    Coaching, Issue, Drill, ISSUE_SUBSCORE_CUTOFF,
    assemble_coaching, build_voice_prompt, is_faithful, generate_coaching,
)
from tennis_cv.types import MetricResult, StrokeScore, StrokeType


def _factor(name, sub_score):
    # value/unit/targets are irrelevant to assembly; only name + sub_score matter.
    return MetricResult(name, 0.0, "u", 0.0, 1.0, sub_score)


def _score(*subscores):
    """Build a StrokeScore from (name, sub_score) pairs; overall = rounded mean."""
    factors = [_factor(n, s) for n, s in subscores]
    overall = round(sum(s for _, s in subscores) / len(subscores)) if subscores else 0
    return StrokeScore(overall=overall, factors=factors)


def test_assemble_flags_below_cutoff_metrics_as_issues_ranked_worst_first():
    score = _score(("knee_flexion", 50), ("contact_height", 100), ("arm_extension", 20))
    coaching = assemble_coaching(StrokeType.SERVE, score)
    titles = [i.title for i in coaching.issues]
    # arm_extension (20) is worst -> priority 1; knee_flexion (50) -> priority 2.
    assert titles == ["Bent arm at contact", "Shallow leg load"]
    assert [i.priority for i in coaching.issues] == [1, 2]
    assert all(len(i.drills) >= 1 for i in coaching.issues)
    assert str(score.overall) in coaching.summary


def test_assemble_with_no_faults_has_no_issues_and_positive_summary():
    score = _score(("knee_flexion", 100), ("contact_height", 90), ("arm_extension", 85))
    coaching = assemble_coaching(StrokeType.SERVE, score)
    assert coaching.issues == []
    assert "no major" in coaching.summary.lower()


def test_cutoff_is_strict():
    # Exactly at the cutoff is a strength, not an issue.
    score = _score(("knee_flexion", ISSUE_SUBSCORE_CUTOFF))
    assert assemble_coaching(StrokeType.SERVE, score).issues == []


def _baseline():
    return assemble_coaching(StrokeType.SERVE, _score(("knee_flexion", 40)))


def test_is_faithful_accepts_prose_only_rewrite():
    base = _baseline()
    issue = base.issues[0]
    reworded = Issue(
        title=issue.title,
        why="A reworded but number-free explanation of the same point.",
        how="A friendlier cue, still number-free.",
        priority=issue.priority,
        drills=issue.drills,  # drills kept verbatim
    )
    # summary reuses only numbers already in the baseline summary.
    rewrite = Coaching(summary=base.summary, issues=[reworded])
    assert is_faithful(base, rewrite) is True


def test_is_faithful_rejects_dropped_issue():
    base = _baseline()
    assert is_faithful(base, Coaching(summary=base.summary, issues=[])) is False


def test_is_faithful_rejects_changed_drill():
    base = _baseline()
    issue = base.issues[0]
    tampered = Issue(
        title=issue.title, why=issue.why, how=issue.how, priority=issue.priority,
        drills=[Drill(name="Totally different drill", description="x")],
    )
    assert is_faithful(base, Coaching(summary=base.summary, issues=[tampered])) is False


def test_is_faithful_rejects_invented_number():
    base = _baseline()
    issue = base.issues[0]
    bad = Issue(
        title=issue.title, why="Improve by 35 percent.", how=issue.how,
        priority=issue.priority, drills=issue.drills,
    )
    assert is_faithful(base, Coaching(summary=base.summary, issues=[bad])) is False


def test_build_voice_prompt_mentions_constraints_and_drills():
    base = _baseline()
    prompt = build_voice_prompt(base, _score(("knee_flexion", 40)))
    assert "Pause-and-drive" in prompt          # a drill name from the baseline
    assert "summary" in prompt and "drills" in prompt


def _faithful_client(baseline):
    # Returns the baseline unchanged -> trivially faithful.
    def parse(**kwargs):
        message = SimpleNamespace(parsed=baseline)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


def _unfaithful_client():
    def parse(**kwargs):
        bad = Coaching(summary="Improve 999 percent", issues=[])
        message = SimpleNamespace(parsed=bad)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


def _raising_client():
    def parse(**kwargs):
        raise RuntimeError("network down")
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


def test_generate_use_voice_false_returns_baseline():
    score = _score(("knee_flexion", 40))
    result = generate_coaching(StrokeType.SERVE, score.factors, score, use_voice=False)
    assert result == assemble_coaching(StrokeType.SERVE, score)


def test_generate_accepts_faithful_voice():
    score = _score(("knee_flexion", 40))
    baseline = assemble_coaching(StrokeType.SERVE, score)
    result = generate_coaching(
        StrokeType.SERVE, score.factors, score, client=_faithful_client(baseline),
    )
    assert result == baseline


def test_generate_rejects_unfaithful_voice_and_falls_back():
    score = _score(("knee_flexion", 40))
    result = generate_coaching(
        StrokeType.SERVE, score.factors, score, client=_unfaithful_client(),
    )
    assert result == assemble_coaching(StrokeType.SERVE, score)


def test_generate_falls_back_when_client_raises():
    score = _score(("knee_flexion", 40))
    result = generate_coaching(
        StrokeType.SERVE, score.factors, score, client=_raising_client(),
    )
    assert result == assemble_coaching(StrokeType.SERVE, score)
