"""Vetted serve drill library.

v1 DRAFT — pending coach review. Drills are drafted from established serve
fundamentals. Review and replace the content via `python -m tennis_cv.preview`
before relying on it with real users. Adding a stroke later = adding entries
here; no code change elsewhere.
"""
from __future__ import annotations

from pydantic import BaseModel


class Drill(BaseModel):
    name: str
    description: str


class FaultEntry(BaseModel):
    metric: str
    title: str
    why: str
    how: str
    drills: list[Drill]


DRILL_LIBRARY: dict[str, FaultEntry] = {
    "knee_flexion": FaultEntry(
        metric="knee_flexion",
        title="Shallow leg load",
        why=(
            "Without a deep knee bend you lose the leg drive that powers an "
            "explosive serve, so the racquet head is slower at contact."
        ),
        how=(
            "Bend your knees more in the trophy position, then drive up and "
            "through the ball as you swing."
        ),
        drills=[
            Drill(
                name="Pause-and-drive",
                description=(
                    "Load into the trophy position, pause for one second to feel "
                    "the knee bend, then explode straight up. Ten reps, no ball."
                ),
            ),
            Drill(
                name="Wall-sit toss",
                description=(
                    "Hold a ninety-degree wall sit for twenty seconds before each "
                    "shadow serve to groove the loaded-leg feeling."
                ),
            ),
        ],
    ),
    "contact_height": FaultEntry(
        metric="contact_height",
        title="Low contact point",
        why=(
            "Striking the ball below full reach lowers your margin over the net "
            "and cuts the downward angle you can hit, costing power and consistency."
        ),
        how=(
            "Reach up and strike the ball at the top of your extension; toss a "
            "little higher and let the ball drop into the top of the strike zone."
        ),
        drills=[
            Drill(
                name="Hit-the-fence-top",
                description=(
                    "Stand near a fence and serve so contact is above the fence "
                    "line, forcing full extension. Fifteen serves."
                ),
            ),
            Drill(
                name="Tall-toss reach",
                description=(
                    "Toss and freeze with your tossing arm fully extended "
                    "overhead; only swing once you feel the stretch upward."
                ),
            ),
        ],
    ),
    "arm_extension": FaultEntry(
        metric="arm_extension",
        title="Bent arm at contact",
        why=(
            "A bent hitting arm at contact shortens your lever and leaks "
            "racquet-head speed, reducing power and making contact inconsistent."
        ),
        how=(
            "Extend your hitting arm so it is nearly straight at contact, reaching "
            "up to the ball rather than pulling down early."
        ),
        drills=[
            Drill(
                name="Straight-arm serves",
                description=(
                    "Serve at half pace focusing only on a fully straight hitting "
                    "arm at contact; build up speed once it feels natural."
                ),
            ),
            Drill(
                name="Trophy-to-contact freeze",
                description=(
                    "Shadow-swing from the trophy position to contact and freeze "
                    "at full extension to check the arm is straight."
                ),
            ),
        ],
    ),
}


def fault_entry(metric: str) -> FaultEntry | None:
    return DRILL_LIBRARY.get(metric)
