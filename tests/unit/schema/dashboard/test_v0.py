from datetime import UTC, date, datetime

import altair as alt
import httpx
import jsonschema
import mandr.schema.dashboard
import pandas as pd
import pytest
import referencing


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
        draft202012validator = jsonschema.Draft202012Validator(
            schema=mandr.schema.dashboard.v0,
            registry=registry,
            format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
        )

        return draft202012validator.validate

    @pytest.mark.parametrize(
        "payload",
        [
            {"key": {"type": "any", "data": None}},
            {"key": {"type": "array", "data": [0, 1, 2]}},
            {"key": {"type": "boolean", "data": True}},
            {"key": {"type": "date", "data": date.today().isoformat()}},
            {"key": {"type": "datetime", "data": datetime.now(tz=UTC).isoformat()}},
            {
                "key": {
                    "type": "file",
                    "data": "file:///tmp/file.txt",
                    "metadata": None,
                    "internal": None,
                }
            },
            {"key": {"type": "html", "data": "<!DOCTYPE html><head></head></html>"}},
            {"key": {"type": "integer", "data": 1}},
            {"key": {"type": "float", "data": 1.2}},
            {"key": {"type": "string", "data": "True"}},
            {
                "key": {
                    "type": "vega",
                    "data": (
                        alt.Chart(pd.DataFrame({"a": ["A"], "b": [28]}))
                        .mark_bar()
                        .to_dict()
                    ),
                }
            },
        ],
    )
    def test_payload_validation(self, validator, payload):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": payload,
            }
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"key": {"type": "array", "data": (0, 1, 2)}},
            {"key": {"type": "boolean", "data": "True"}},
            {"key": {"type": "date", "data": datetime.now(tz=UTC).isoformat()}},
            {"key": {"type": "datetime", "data": date.today().isoformat()}},
            {
                "key": {
                    "type": "file",
                    "data": "file.txt",
                    "metadata": None,
                    "internal": None,
                }
            },
            {"key": {"type": "html", "data": "<head></head>"}},
            {"key": {"type": "integer", "data": 1.2}},
            {"key": {"type": "float", "data": "1.2"}},
            {"key": {"type": "string", "data": True}},
            {"key": {"type": "vega", "data": None}},
        ],
    )
    def test_payload_invalidation(self, validator, payload):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": payload,
                }
            )
