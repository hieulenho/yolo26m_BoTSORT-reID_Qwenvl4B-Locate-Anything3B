import argparse

import pytest

from football_tracking.cli import _build_parser


def test_public_cli_is_dataset_independent() -> None:
    parser = _build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert {'doctor', 'track-video', 'compare-trackers', 'evaluate-tracking'} <= set(action.choices)
    assert not any('sportsmot' in command or 'team' in command for command in action.choices)


def test_tracking_requires_an_explicit_config() -> None:
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(['track-video'])
    assert error.value.code == 2
