import pytest
from unittest.mock import MagicMock
from skore import Project, EstimatorReport

class TestProjectDictInterface:
    @pytest.fixture
    def mock_plugin(self):
        """Create a mock plugin to avoid real disk I/O."""
        return MagicMock()

    @pytest.fixture
    def project(self, mock_plugin):
        """Create a Project instance with the mock plugin injected."""
        # Necessary to bypass the standard __init__ to inject our mock
        project = Project.__new__(Project)
        project._Project__project = mock_plugin
        project._Project__mode = "local"
        project._Project__name = "test_project"
        return project

    def test_setitem_calls_put(self, project, mock_plugin):
        """Test project['key'] = report calls put()."""
        report = MagicMock(spec=EstimatorReport)
        report.ml_task = "regression"
        # Mock ml_task to avoid validation error
        project.ml_task = "regression"
        
        project["my_key"] = report
        
        mock_plugin.put.assert_called_once_with(key="my_key", report=report)

    def test_getitem_calls_get(self, project, mock_plugin):
        """Test project['key'] retrieves the item via summary lookup."""

        mock_plugin.summarize.return_value = [
            {"key": "my_key", "id": "123", "ml_task": "regression"}
        ]
        
        _ = project["my_key"]
        
        mock_plugin.get.assert_called_once_with("123")

    def test_delitem_calls_delete_item(self, project, mock_plugin):
        """Test del project['key'] calls delete_item()."""
        del project["my_key"]
        mock_plugin.delete_item.assert_called_once_with("my_key")

    def test_contains(self, project, mock_plugin):
        """Test 'key' in project."""
        mock_plugin.summarize.return_value = [
            {"key": "present_key", "id": "123", "ml_task": "regression"}
        ]
        
        assert "present_key" in project
        assert "missing_key" not in project