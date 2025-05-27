def check_display_data(display):
    """Check the structure of the display's internal data."""
    assert list(display.precision_recall.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "precision",
        "recall",
    ]
    assert list(display.average_precision.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "average_precision",
    ]
