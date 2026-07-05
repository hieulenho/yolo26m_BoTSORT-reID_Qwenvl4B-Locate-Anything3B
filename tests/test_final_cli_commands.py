from __future__ import annotations

import argparse

from football_tracking.cli import _build_parser


def test_final_milestone_commands_are_registered() -> None:
    parser = _build_parser()
    subparser_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )

    commands = set(subparser_action.choices)

    assert {
        "doctor",
        "prepare-dataset",
        "train-detector",
        "evaluate-detector",
        "build-domain-configs",
        "cache-detections",
        "track",
        "compare-trackers",
        "evaluate-tracking",
        "plan-tracker-grid",
        "render-video",
        "analyze-tracking-vlm",
        "benchmark",
        "generate-report",
        "summarize-experiments",
    }.issubset(commands)
