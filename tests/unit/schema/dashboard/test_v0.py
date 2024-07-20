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

    def test_schema_with_any(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "any", "data": None}},
            }
        )

    def test_schema_with_array(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "array", "data": [0, 1, 2]}},
            }
        )

    def test_schema_with_array_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": {"type": "array", "data": (0, 1, 2)}},
                }
            )

    def test_schema_with_boolean(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "boolean", "data": True}},
            }
        )

    def test_schema_with_boolean_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": {"type": "boolean", "data": "True"}},
                }
            )

    def test_schema_with_date(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "date", "data": date.today().isoformat()}},
            }
        )

    def test_schema_with_date_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {
                        "key": {
                            "type": "date",
                            "data": datetime.now(tz=UTC).isoformat(),
                        }
                    },
                }
            )

    def test_schema_with_datetime(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {
                    "key": {
                        "type": "datetime",
                        "data": datetime.now(tz=UTC).isoformat(),
                    }
                },
            }
        )

    def test_schema_with_datetime_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {
                        "key": {"type": "datetime", "data": date.today().isoformat()}
                    },
                }
            )

    def test_schema_with_file(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {
                    "key": {
                        "type": "file",
                        "data": "file:///tmp/file.txt",
                        "metadata": None,
                        "internal": None,
                    }
                },
            }
        )

    def test_schema_with_file_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {
                        "key": {
                            "type": "file",
                            "data": "file.txt",
                            "metadata": None,
                            "internal": None,
                        }
                    },
                }
            )

    def test_schema_with_html(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {
                    "key": {
                        "type": "html",
                        "data": "<!DOCTYPE html><head></head></html>",
                    }
                },
            }
        )

    def test_schema_with_html_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": {"type": "html", "data": "<head></head>"}},
                }
            )

    def test_schema_with_integer(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "integer", "data": 1}},
            }
        )

    def test_schema_with_integer_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": {"type": "integer", "data": 1.2}},
                }
            )

    def test_schema_with_float(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "float", "data": 1.2}},
            }
        )

    def test_schema_with_float_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": {"type": "float", "data": "1.2"}},
                }
            )

    def test_schema_with_string(self, validator):
        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "string", "data": "True"}},
            }
        )

    def test_schema_with_string_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": {"type": "string", "data": True}},
                }
            )

    def test_schema_with_vega(self, validator):
        source = pd.DataFrame({"a": ["A"], "b": [28]})
        chart = alt.Chart(source).mark_bar().encode(x="a", y="b")

        validator(
            instance={
                "schema": "schema:dashboard:v0",
                "uri": "test",
                "payload": {"key": {"type": "vega", "data": chart.to_dict()}},
            }
        )

    def test_schema_with_vega_exception(self, validator):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validator(
                instance={
                    "schema": "schema:dashboard:v0",
                    "uri": "test",
                    "payload": {"key": {"type": "vega", "data": None}},
                }
            )
