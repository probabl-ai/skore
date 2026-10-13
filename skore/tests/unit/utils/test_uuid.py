from uuid import uuid1, uuid4

import pytest

from skore._utils.uuid import normalize_report_id, uuid7


def test_normalize_report_id_accepts_uuid_forms():
    """Report IDs are accepted as UUID objects, integers, and strings."""
    report_id = uuid4()

    assert normalize_report_id(report_id) == str(report_id)
    assert normalize_report_id(report_id.int) == str(report_id)
    assert normalize_report_id(str(report_id.int)) == str(report_id)
    assert normalize_report_id(str(report_id)) == str(report_id)

    report_id = uuid7()
    assert normalize_report_id(report_id) == str(report_id)


def test_normalize_report_id_rejects_invalid_values():
    """Booleans, other types, and non-v4/v7 IDs are rejected."""
    with pytest.raises(TypeError, match="Invalid report ID type"):
        normalize_report_id(True)
    with pytest.raises(TypeError, match="Invalid report ID type"):
        normalize_report_id(1.5)
    with pytest.raises(ValueError, match="Unsupported report ID"):
        normalize_report_id(uuid1())
