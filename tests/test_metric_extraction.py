from football_tracking.detection.metrics import parse_ultralytics_metrics


class _Box:
    mp = 0.11
    mr = 0.22
    map50 = 0.33
    map = 0.44
    map75 = 0.55


class _Metrics:
    box = _Box()


def test_metric_extraction_does_not_swap_map_fields() -> None:
    metrics = parse_ultralytics_metrics(_Metrics())

    assert metrics.map50 == 0.33
    assert metrics.map50_95 == 0.44
    assert metrics.map75 == 0.55


def test_metric_extraction_missing_metrics_are_null() -> None:
    metrics = parse_ultralytics_metrics(object())

    assert metrics.map50 is None
    assert metrics.reason
