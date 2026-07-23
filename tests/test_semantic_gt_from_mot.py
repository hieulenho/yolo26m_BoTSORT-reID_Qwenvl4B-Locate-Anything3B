from __future__ import annotations

from pathlib import Path

from scripts.benchmarks.build_semantic_gt_from_mot import _categories


def test_categories_support_flat_and_hierarchical_labels(tmp_path: Path) -> None:
    path = tmp_path / "categories.yaml"
    path.write_text(
        """
1: car
2:
  class_label: car
  fine_label: van
3:
  class_label: unknown
""",
        encoding="utf-8",
    )

    assert _categories(path) == {
        1: {"class_label": "car", "fine_label": ""},
        2: {"class_label": "car", "fine_label": "van"},
        3: {"class_label": "unknown", "fine_label": ""},
    }


def test_categories_reject_hierarchical_row_without_base_label(tmp_path: Path) -> None:
    path = tmp_path / "categories.yaml"
    path.write_text("1:\n  fine_label: van\n", encoding="utf-8")

    try:
        _categories(path)
    except ValueError as exc:
        assert "requires class_label" in str(exc)
    else:
        raise AssertionError("Expected missing class_label to fail.")
