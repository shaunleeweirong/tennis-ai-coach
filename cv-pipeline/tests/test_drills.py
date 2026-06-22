from tennis_cv.drills import Drill, FaultEntry, DRILL_LIBRARY, fault_entry

# These are exactly the metric names emitted by tennis_cv.metrics.compute_serve_metrics.
SERVE_METRICS = {"knee_flexion", "contact_height", "arm_extension"}


def test_library_covers_every_serve_metric():
    assert SERVE_METRICS <= set(DRILL_LIBRARY)


def test_every_entry_is_well_formed():
    for metric, entry in DRILL_LIBRARY.items():
        assert isinstance(entry, FaultEntry)
        assert entry.metric == metric
        assert entry.title.strip()
        assert entry.why.strip()
        assert entry.how.strip()
        assert len(entry.drills) >= 1
        for drill in entry.drills:
            assert isinstance(drill, Drill)
            assert drill.name.strip()
            assert drill.description.strip()


def test_fault_entry_lookup():
    assert fault_entry("knee_flexion").title == DRILL_LIBRARY["knee_flexion"].title
    assert fault_entry("not_a_metric") is None
