from contextlib import nullcontext as does_not_raise
from datetime import UTC, date, datetime

import altair as alt
import httpx
import mandr.schema.dashboard
import pandas as pd
import pytest
import referencing
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


class TestV0:
    @pytest.fixture(scope="class")
    def validator(self):
        """
        Validator with format checking and external schemas pulling.

        Notes
        -----
        Check if primitive types conform to well-defined formats.
            https://python-jsonschema.readthedocs.io/en/latest/validate/#validating-formats

        Automatically retrieve referenced schemas on-the-fly:
            https://python-jsonschema.readthedocs.io/en/stable/referencing/#automatically-retrieving-resources-over-http
        """

        def retrieve_via_httpx(uri):
            response = httpx.get(uri)
            return referencing.Resource.from_contents(response.json())

        registry = referencing.Registry(retrieve=retrieve_via_httpx)
        draft202012validator = Draft202012Validator(
            schema=mandr.schema.dashboard.v0,
            registry=registry,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )

        return draft202012validator.validate

    @pytest.mark.parametrize(
        "payload,expectation",
        [
            (
                {"type": "unknown", "data": None},
                pytest.raises(
                    ValidationError, match="Failed validating 'enum' in schema"
                ),
            ),
            ({"type": "any", "data": None}, does_not_raise()),
            ({"type": "array", "data": [0, 1, 2]}, does_not_raise()),
            (
                {"type": "array", "data": (0, 1, 2)},
                pytest.raises(ValidationError, match="is not of type 'array'"),
            ),
            ({"type": "boolean", "data": True}, does_not_raise()),
            (
                {"type": "boolean", "data": "True"},
                pytest.raises(ValidationError, match="is not of type 'boolean'"),
            ),
            ({"type": "date", "data": date.today().isoformat()}, does_not_raise()),
            (
                {"type": "date", "data": datetime.now(tz=UTC).isoformat()},
                pytest.raises(ValidationError, match="is not a 'date'"),
            ),
            (
                {"type": "datetime", "data": datetime.now(tz=UTC).isoformat()},
                does_not_raise(),
            ),
            (
                {"type": "datetime", "data": date.today().isoformat()},
                pytest.raises(ValidationError, match="is not a 'date-time'"),
            ),
            (
                {
                    "type": "file",
                    "data": "file:///tmp/file.txt",
                    "metadata": None,
                    "internal": None,
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "file",
                    "data": "file.txt",
                    "metadata": None,
                    "internal": None,
                },
                pytest.raises(ValidationError, match="is not a 'uri'"),
            ),
            (
                {"type": "html", "data": "<!DOCTYPE html><head></head></html>"},
                does_not_raise(),
            ),
            (
                {"type": "html", "data": "<head></head>"},
                pytest.raises(ValidationError, match="Failed validating 'pattern'"),
            ),
            ({"type": "integer", "data": 1}, does_not_raise()),
            (
                {"type": "integer", "data": 1.2},
                pytest.raises(ValidationError, match="is not of type 'integer'"),
            ),
            ({"type": "number", "data": 1.2}, does_not_raise()),
            (
                {"type": "number", "data": "1.2"},
                pytest.raises(ValidationError, match="is not of type 'number'"),
            ),
            ({"type": "string", "data": "True"}, does_not_raise()),
            (
                {"type": "string", "data": True},
                pytest.raises(ValidationError, match="is not of type 'string'"),
            ),
            (
                {
                    "type": "vega",
                    "data": (
                        alt.Chart(pd.DataFrame({"a": ["A"], "b": [28]}))
                        .mark_bar()
                        .to_dict()
                    ),
                },
                does_not_raise(),
            ),
            (
                {"type": "vega", "data": None},
                pytest.raises(
                    ValidationError, match="is not valid under any of the given schemas"
                ),
            ),
        ],
    )
    def test_payload_validation(self, validator, payload, expectation):
        with expectation:
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": payload},
                }
            )
