import pytest
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from skore import EstimatorReport
from skore_local_project.project import Project

class TestLocalProjectDeletion:
    @pytest.fixture
    def mock_report(self):
        """Create a valid EstimatorReport for testing."""
        model = LinearRegression()
        X = pd.DataFrame({"a": [0]})
        y = pd.Series([0])
        model.fit(X, y)
        return EstimatorReport(model, X_train=X, y_train=y, X_test=X, y_test=y)

    def test_delete_item_removes_metadata_and_artifact(self, tmp_path, mock_report):
        """Test that deleting an item removes its metadata and the artifact if unused."""
        # 1. Initiate
        project = Project("test_proj", workspace=tmp_path)
        
        # 2. Put item
        project.put("item1", mock_report)
        
        # Checking it exists
        summary = project.summarize()
        assert len(summary) == 1
        assert summary[0]["key"] == "item1"
        
        # 3. Item Deletion
        project.delete_item("item1")
        
        # 4. Verify Metadata is gone
        assert len(project.summarize()) == 0
        
        # 5. Verifying Artifact is garbage collected
        assert len(project._Project__artifacts_storage) == 0

    def test_delete_item_keeps_artifact_if_shared(self, tmp_path, mock_report):
        """Test that artifact is NOT deleted if another key uses it."""
        project = Project("test_proj", workspace=tmp_path)
        
        # Putting same report under two keys
        project.put("key1", mock_report)
        project.put("key2", mock_report)
        
        # Artifact count should be 1 (deduplication)
        assert len(project._Project__artifacts_storage) == 1
        
        # Delete one key
        project.delete_item("key1")
        
        # Artifact should STILL exist because key2 needs it
        assert len(project._Project__artifacts_storage) == 1
        
        # Delete second key
        project.delete_item("key2")
        
        # Now artifact should be gone
        assert len(project._Project__artifacts_storage) == 0