from __future__ import annotations

from football_tracking.benchmarking.final_report import _validate_idsw


def test_idsw_taxonomy_accepts_complete_category_partition() -> None:
    payload = {
        "summaries": [
            {
                "sequence": "__overall__",
                "tracker": "fasttrack",
                "total_id_switches_recomputed": 10,
                "fragmentation_count": 2,
                "fragmentation_percent": 20.0,
                "identity_swap_count": 2,
                "identity_swap_percent": 20.0,
                "re_identification_failure_count": 2,
                "re_identification_failure_percent": 20.0,
                "association_error_count": 2,
                "association_error_percent": 20.0,
                "appearance_confusion_count": 2,
                "appearance_confusion_percent": 20.0,
            }
        ]
    }
    issues: list[dict] = []

    _validate_idsw(payload, [{"tracker": "fasttrack"}], issues)

    assert issues == []


def test_idsw_taxonomy_rejects_incomplete_partition() -> None:
    payload = {
        "summaries": [
            {
                "sequence": "__overall__",
                "tracker": "fasttrack",
                "total_id_switches_recomputed": 10,
                "fragmentation_count": 1,
                "fragmentation_percent": 10.0,
            }
        ]
    }
    issues: list[dict] = []

    _validate_idsw(payload, [{"tracker": "fasttrack"}], issues)

    assert {issue["code"] for issue in issues} == {"idsw_count_sum", "idsw_percent_sum"}
