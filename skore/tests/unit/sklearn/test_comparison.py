import pandas as pd


def test_comparison_timings_flat_index(multi_estimate_binary_report):
    """Check the behaviour of the `timings` method with flat_index=True."""
    report = multi_estimate_binary_report

    report.metrics.report_metrics(data_source="train")
    report.metrics.report_metrics(data_source="test")

    timings = report.metrics.timings(flat_index=True)
    assert isinstance(timings, pd.DataFrame)

    assert all("_s" in idx for idx in timings.index if "time" in idx.lower())
    assert all("(s)" not in idx for idx in timings.index)

    results = report.metrics.report_metrics(flat_index=True)

    time_indices = [idx for idx in results.index if "time" in idx]
    for idx in time_indices:
        assert idx.endswith("_s")
        assert "(s)" not in idx
