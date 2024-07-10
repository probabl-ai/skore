from operator import attrgetter

import joblib
import pytest
from mandr import InfoMander


class TestInfoMander:
    def test_check_key(self):
        for key in InfoMander.RESERVED_KEYS:
            with pytest.raises(ValueError):
                InfoMander._check_key(key)

    def test_add_info_overwrite_true(self, mock_nowstr, mock_mandr):
        mock_mandr.add_info("key1", "value1", overwrite=True)
        mock_mandr.add_info("key2", "value1", overwrite=True)
        mock_mandr.add_info("key2", "value2", overwrite=True)

        assert mock_mandr.cache == {
            "artifacts": {},
            "templates": {},
            "views": {},
            "logs": {},
            "key1": "value1",
            "key2": "value2",
            "updated_at": mock_nowstr,
        }

    def test_add_info_overwrite_false(self, mock_nowstr, mock_mandr):
        mock_mandr.add_info("key1", "value1", overwrite=False)
        mock_mandr.add_info("key2", ["value1"], overwrite=False)
        mock_mandr.add_info("key2", ["value2"], overwrite=False)
        mock_mandr.add_info("key3", "value1", overwrite=False)
        mock_mandr.add_info("key3", "value2", overwrite=False)

        assert mock_mandr.cache == {
            "artifacts": {},
            "templates": {},
            "views": {},
            "logs": {},
            "key1": ["value1"],
            "key2": ["value2", "value1"],
            "key3": ["value2", "value1"],
            "updated_at": mock_nowstr,
        }

    def test_add_to_key(self, mock_nowstr, mock_mandr):
        mock_mandr._add_to_key("artifacts", "key", "value")

        assert mock_mandr.cache == {
            "artifacts": {"key": "value"},
            "templates": {},
            "views": {},
            "logs": {},
            "updated_at": mock_nowstr,
        }

    def test_add_artifact(self, mock_now, mock_nowstr, mock_mandr, tmp_path):
        mock_mandr.add_artifact("key", "value", datetime=mock_now)

        assert joblib.load(tmp_path / "root" / ".artifacts" / "key.joblib") == "value"
        assert mock_mandr.cache == {
            "artifacts": {"key": {"datetime": mock_now}},
            "templates": {},
            "views": {},
            "logs": {},
            "updated_at": mock_nowstr,
        }

    def test_add_view(self, mock_nowstr, mock_mandr):
        mock_mandr.add_view("key", "<!DOCTYPE html><html>hydrated</html>")

        assert mock_mandr.cache == {
            "artifacts": {},
            "templates": {},
            "views": {"key": "<!DOCTYPE html><html>hydrated</html>"},
            "logs": {},
            "updated_at": mock_nowstr,
        }

    def test_add_template(self, mock_nowstr, mock_mandr):
        mock_mandr.add_template("key", "<!DOCTYPE html><html>{{template}}</html>")

        assert mock_mandr.cache == {
            "artifacts": {},
            "templates": {"key": "<!DOCTYPE html><html>{{template}}</html>"},
            "views": {},
            "logs": {},
            "updated_at": mock_nowstr,
        }

    def test_add_template_exception(self, mock_mandr):
        mock_mandr.cache["views"] = {"key": "<!DOCTYPE html><html>hydrated</html>"}

        with pytest.raises(ValueError):
            mock_mandr.add_template("key", "<!DOCTYPE html><html>{{template}}</html>")

    def test_render_templates(self, mock_nowstr, mock_mandr):
        class MockTemplate:
            def render(self, *args, **kwargs):
                return "<!DOCTYPE html><html>hydrated</html>"

        template = MockTemplate()
        mock_mandr.cache["templates"] = {"key": template}
        mock_mandr.render_templates()

        assert mock_mandr.cache == {
            "artifacts": {},
            "templates": {"key": template},
            "views": {"key": "<!DOCTYPE html><html>hydrated</html>"},
            "logs": {},
            "updated_at": mock_nowstr,
        }

    def test_add_logs(self, mock_nowstr, mock_mandr):
        mock_mandr.add_logs("key", "value")

        assert mock_mandr.cache == {
            "artifacts": {},
            "templates": {},
            "views": {},
            "logs": {"key": "value"},
            "updated_at": mock_nowstr,
        }

    def test_fetch(self, mock_mandr):
        assert mock_mandr.fetch() == {
            "artifacts": {},
            "templates": {},
            "views": {},
            "logs": {},
        }

    def test_getitem(self, mock_mandr):
        assert mock_mandr["artifacts"] == {}

    def test_dsl_path_exists(self, mock_mandr, tmp_path):
        (tmp_path / "root2").mkdir(parents=True)
        (tmp_path / "root3").mkdir(parents=True)

        assert mock_mandr.dsl_path_exists("root2") is None
        assert mock_mandr.dsl_path_exists("root3") is None

        with pytest.raises(AssertionError):
            mock_mandr.dsl_path_exists("root4")

    def test_child(self, mock_mandr, tmp_path):
        tmp_path /= "root"

        (tmp_path).mkdir(parents=True)
        (tmp_path / "subroot1").mkdir(parents=True)

        assert mock_mandr.get_child("subroot1") == InfoMander("subroot1", root=tmp_path)
        assert mock_mandr.get_child("subroot2") == InfoMander("subroot2", root=tmp_path)

    def test_children(self, mock_mandr, tmp_path):
        tmp_path /= "root"

        (tmp_path).mkdir(parents=True)
        (tmp_path / ".artifacts").mkdir(parents=True)
        (tmp_path / ".stats").mkdir(parents=True)

        assert mock_mandr.children() == []

        (tmp_path / "subroot1").mkdir(parents=True)
        (tmp_path / "subroot2").mkdir(parents=True)

        assert sorted(mock_mandr.children(), key=attrgetter("project_path")) == [
            InfoMander("subroot1", root=tmp_path),
            InfoMander("subroot2", root=tmp_path),
        ]

    def test_repr(self, mock_mandr, tmp_path):
        assert repr(mock_mandr) == f"InfoMander({tmp_path / 'root'})"

    def test_eq(self, tmp_path):
        IM = InfoMander

        assert IM("root", root=tmp_path) == IM("root", root=tmp_path)
        assert IM("root/subroot", root=tmp_path) == IM("root/subroot", root=tmp_path)
        assert IM("root1", root=tmp_path) != IM("root2", root=tmp_path)
        assert IM("root1/subroot", root=tmp_path) != IM("root2/subroot", root=tmp_path)
        assert IM("root1/subroot1", root=tmp_path) != IM(
            "root1/subroot2", root=tmp_path
        )
