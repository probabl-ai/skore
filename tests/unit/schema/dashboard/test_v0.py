import json
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
            (
                {
                    "type": "cv_results",
                    "data": {
                        "cv_results_table": {
                            "type": "dataframe",
                            "data": json.loads(
                                pd.DataFrame(
                                    [[1, 2, 3], [4, 5, 6]],
                                    columns=["column_0", 1, True],
                                ).to_json(orient="table")
                            ),
                        },
                        "roc_curve_spec": {
                            "type": "vega",
                            "data": (
                                alt.Chart(pd.DataFrame({"a": ["A"], "b": [28]}))
                                .mark_bar()
                                .to_dict()
                            ),
                        },
                    },
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "cv_results",
                    "data": {
                        "cv_results_table": json.loads(
                            pd.DataFrame(
                                [[1, 2, 3], [4, 5, 6]],
                                columns=["column_0", 1, True],
                            ).to_json(orient="table")
                        ),
                        "roc_curve_spec": (
                            alt.Chart(pd.DataFrame({"a": ["A"], "b": [28]}))
                            .mark_bar()
                            .to_dict()
                        ),
                    },
                },
                # The error message is not particularly informative.
                # The issue is we need to keep using the "type" and "data" keys even
                # when nested.
                pytest.raises(
                    ValidationError, match="not valid under any of the given schemas"
                ),
            ),
            (
                {
                    "type": "dataframe",
                    "data": json.loads(
                        pd.DataFrame(
                            [[1, 2, 3], [4, 5, 6]],
                            columns=["column_0", 1, True],
                        ).to_json(orient="table")
                    ),
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "dataframe",
                    "data": {
                        "schema": {
                            "fields": [
                                {"name": "index", "type": "integer"},
                                {"name": "column_0", "type": "integer"},
                                {"name": 1, "type": "integer"},
                                {"name": True, "type": "integer"},
                            ],
                            "primaryKey": ["index"],
                        },
                        # Note the data is inconsistent with the fields
                        "data": [
                            {"hello": 3},
                            {"goodbye": "hey"},
                        ],
                    },
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "dataframe",
                    "data": json.loads(
                        pd.DataFrame(
                            [[1, 2, 3], [4, 5, 6]],
                            columns=["column_0", 1, True],
                        ).to_json(orient="split")
                    ),
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "dataframe",
                    # to_json outputs a string, not a dict
                    "data": pd.DataFrame(
                        [[1, 2, 3], [4, 5, 6]],
                        columns=["column_0", 1, True],
                    ).to_json(orient="table"),
                },
                pytest.raises(ValidationError, match="is not of type 'object'"),
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
                {
                    "type": "image",
                    "data": {
                        "mime-type": "image/svg+xml",
                        "data": (
                            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZ"
                            "pZXdCb3g9IjAgMCAxMDAgMTAwIj4KICA8Y2lyY2xlIGN4PSI1MCIgY3"
                            "k9IjUwIiByPSI0OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMDAwIi8+C"
                            "iAgPHBhdGggZD0iTTUwLDJhNDgsNDggMCAxIDEgMCw5NmEyNCAyNCAw"
                            "IDEgMSAwLTQ4YTI0IDI0IDAgMSAwIDAtNDgiLz4KICA8Y2lyY2xlIGN"
                            "4PSI1MCIgY3k9IjI2IiByPSI2Ii8+CiAgPGNpcmNsZSBjeD0iNTAiIG"
                            "N5PSI3NCIgcj0iNiIgZmlsbD0iI0ZGRiIvPgo8L3N2Zz4="
                        ),
                    },
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "image",
                    "data": {
                        "mime-type": "image/webp",
                        "data": "lkahsi34982djGGD/=",
                    },
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "image",
                    # NOTE: Invalid Base64
                    "data": {"mime-type": "image/png", "data": "hello.22"},
                },
                pytest.raises(ValidationError, match="Failed validating 'pattern'"),
            ),
            (
                {"type": "image", "data": "<!DOCTYPE svg></svg>"},
                pytest.raises(ValidationError, match="is not of type 'object'"),
            ),
            (
                {
                    "type": "image",
                    "data": {
                        "mime-type": "image/svg",
                        "data": "<!DOCTYPE svg></svg>",
                    },
                },
                pytest.raises(ValidationError, match="Failed validating 'enum'"),
            ),
            (
                {"type": "html", "data": "<!DOCTYPE html><head></head></html>"},
                does_not_raise(),
            ),
            (
                {"type": "html", "data": "<head></head>"},
                # NOTE: No DOCTYPE
                does_not_raise(),
            ),
            ({"type": "integer", "data": 1}, does_not_raise()),
            (
                {"type": "integer", "data": 1.2},
                pytest.raises(ValidationError, match="is not of type 'integer'"),
            ),
            (
                {
                    "type": "markdown",
                    "data": "# Hello\n## Hi\n\nThis is a markdown string",
                },
                does_not_raise(),
            ),
            (
                {
                    "type": "markdown",
                    "data": "<p>This is technically valid Markdown</p>",
                },
                does_not_raise(),
            ),
            (
                {"type": "markdown", "data": 1},
                pytest.raises(ValidationError, match="is not of type 'string'"),
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
