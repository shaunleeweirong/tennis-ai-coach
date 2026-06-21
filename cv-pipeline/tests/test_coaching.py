from types import SimpleNamespace
from tennis_cv.coaching import build_prompt, generate_coaching, Coaching, Issue, Drill
from tennis_cv.types import MetricResult, StrokeScore, StrokeType


def _score():
    factors = [
        MetricResult("knee_flexion", 95, "deg", 110, 140, 50.0),
        MetricResult("contact_height", 2.0, "ratio", 1.5, 2.5, 100.0),
        MetricResult("arm_extension", 150, "deg", 160, 180, 0.0),
    ]
    return factors, StrokeScore(overall=50, factors=factors)


def test_build_prompt_includes_metric_values_and_targets():
    factors, score = _score()
    prompt = build_prompt(StrokeType.SERVE, factors, score)
    assert "knee_flexion" in prompt
    assert "95" in prompt
    assert "110" in prompt and "140" in prompt
    assert "50" in prompt  # overall score


def test_generate_coaching_uses_injected_client_and_returns_parsed():
    factors, score = _score()
    expected = Coaching(
        summary="Drive your legs more and finish the reach.",
        issues=[Issue(title="Shallow knee bend", why="Less power.", how="Bend deeper.", priority=1)],
        drills=[Drill(name="Pause-and-drive", description="Load, pause, explode up.", addresses="knee_flexion")],
    )

    captured = {}

    def fake_parse(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(parsed=expected)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=fake_parse))
    )

    result = generate_coaching(StrokeType.SERVE, factors, score, client=fake_client)

    assert result == expected
    assert captured["model"] == "gemini-2.5-flash-lite"
    assert captured["response_format"] is Coaching
