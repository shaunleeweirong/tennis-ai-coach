from tennis_cv.preview import REPRESENTATIVE_PROFILES, render_profile, main


def test_profiles_cover_the_expected_cases():
    names = [name for name, _ in REPRESENTATIVE_PROFILES]
    assert "all-good" in names
    assert any("multi" in n for n in names)
    assert len(REPRESENTATIVE_PROFILES) >= 5


def test_render_profile_deterministic_contains_drill_for_a_fault():
    # Find the shallow-knee profile and confirm its rendered text names a knee drill.
    name, score = next(p for p in REPRESENTATIVE_PROFILES if p[0] == "shallow-knee")
    text = render_profile(name, score, use_voice=False)
    assert "Pause-and-drive" in text
    assert "Shallow leg load" in text


def test_main_renders_all_profiles(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    for name, _ in REPRESENTATIVE_PROFILES:
        assert name in out
