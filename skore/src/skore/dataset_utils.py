"""Utilities for dataset comparison and validation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings
from scipy import stats


def compare_datasets(
    dataset1: Union[pd.DataFrame, np.ndarray],
    dataset2: Union[pd.DataFrame, np.ndarray],
    id_column: Optional[str] = None,
    verbose: bool = True,
    feature_distribution: bool = False,
) -> Dict[str, Any]:
    """
    Compare two datasets and return a summary of differences to detect potential data leakage.
    
    This function helps identify overlapping data between training and evaluation datasets,
    which could lead to overoptimistic model performance estimates.
    
    Parameters
    ----------
    dataset1 : DataFrame or array
        First dataset for comparison (e.g., training data)
    dataset2 : DataFrame or array
        Second dataset for comparison (e.g., test data)
    id_column : str, optional
        Column name to use for identifying unique rows. If provided and present in both
        datasets, will be used to find overlapping records.
    verbose : bool, default=True
        Whether to print comparison summary
    feature_distribution : bool, default=False
        Whether to compare statistical distributions of shared numerical features
        
    Returns
    -------
    Dict
        Dictionary containing comparison results with the following keys:
        - shape_comparison: Information about dataset dimensions
        - column_comparison: Analysis of columns in both datasets
        - overlap_analysis: Information about overlapping data points
        - feature_distribution: Statistical tests comparing distributions (if requested)
        
    Examples
    --------
    >>> import pandas as pd
    >>> from skore.dataset_utils import compare_datasets
    >>> 
    >>> # Basic comparison with ID columns
    >>> train_df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    >>> test_df = pd.DataFrame({"id": [3, 4, 5], "value": [30, 40, 50]})
    >>> result = compare_datasets(train_df, test_df, id_column="id")
    
    >>> # Compare data used with report_metrics to check for potential data leakage
    >>> from skore import EstimatorReport
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> # First create and train a model with training data
    >>> X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    >>> y_train = [0, 0, 1]
    >>> X_new = pd.DataFrame({"feature1": [2, 3, 4], "feature2": [5, 6, 7]})
    >>> y_new = [0, 1, 1]
    >>> 
    >>> # Check if "new" data overlaps with training data before using it
    >>> leakage_check = compare_datasets(X_train, X_new)
    >>> 
    >>> # Only use for evaluation if overlap is minimal
    >>> model = RandomForestClassifier()
    >>> report = EstimatorReport(model, X_train=X_train, y_train=y_train)
    >>> if leakage_check["overlap_analysis"].get("overlap_percentage", 0) < 10:
    >>>     report.report_metrics(X_y=(X_new, y_new))
    >>> else:
    >>>     print("Warning: Significant overlap detected with training data")
    """
    # Convert numpy arrays to pandas DataFrames if needed
    if isinstance(dataset1, np.ndarray):
        dataset1 = pd.DataFrame(dataset1)
    if isinstance(dataset2, np.ndarray):
        dataset2 = pd.DataFrame(dataset2)
    
    # Initialize results dictionary
    results = {
        "shape_comparison": {
            "dataset1_shape": dataset1.shape,
            "dataset2_shape": dataset2.shape,
            "same_shape": dataset1.shape == dataset2.shape
        },
        "column_comparison": {},
        "overlap_analysis": {}
    }
    
    # Compare columns
    dataset1_cols = set(dataset1.columns)
    dataset2_cols = set(dataset2.columns)
    
    results["column_comparison"] = {
        "shared_columns": list(dataset1_cols.intersection(dataset2_cols)),
        "only_in_dataset1": list(dataset1_cols - dataset2_cols),
        "only_in_dataset2": list(dataset2_cols - dataset1_cols),
        "same_columns": dataset1_cols == dataset2_cols
    }
    
    # Check for overlapping data
    if id_column and id_column in dataset1.columns and id_column in dataset2.columns:
        # Using specified ID column
        dataset1_ids = set(dataset1[id_column])
        dataset2_ids = set(dataset2[id_column])
        
        overlap_ids = dataset1_ids.intersection(dataset2_ids)
        results["overlap_analysis"] = {
            "overlapping_ids": list(overlap_ids),
            "overlap_count": len(overlap_ids),
            "overlap_percentage": round(len(overlap_ids) / len(dataset1_ids) * 100, 2)
        }
    else:
        # Try to check for duplicate rows
        try:
            shared_columns = results["column_comparison"]["shared_columns"]
            if shared_columns:
                # Check overlapping rows based on shared columns
                dataset1_subset = dataset1[shared_columns].drop_duplicates()
                dataset2_subset = dataset2[shared_columns].drop_duplicates()
                
                merged = pd.merge(
                    dataset1_subset, dataset2_subset, 
                    on=shared_columns, how='inner'
                )
                
                overlap_count = len(merged)
                results["overlap_analysis"] = {
                    "overlap_row_count": overlap_count,
                    "overlap_percentage": round(overlap_count / len(dataset1) * 100, 2)
                }
            else:
                results["overlap_analysis"] = {
                    "message": "No shared columns to check for overlapping rows"
                }
        except Exception as e:
            results["overlap_analysis"] = {
                "message": f"Could not analyze overlap: {str(e)}"
            }
    
    # Compare feature distributions if requested
    if feature_distribution:
        results["feature_distribution"] = {}
        shared_columns = results["column_comparison"]["shared_columns"]
        
        for col in shared_columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(dataset1[col]) and pd.api.types.is_numeric_dtype(dataset2[col]):
                try:
                    # Perform Kolmogorov-Smirnov test for distribution comparison
                    ks_stat, p_value = stats.ks_2samp(
                        dataset1[col].dropna().values, 
                        dataset2[col].dropna().values
                    )
                    
                    results["feature_distribution"][col] = {
                        "ks_statistic": ks_stat,
                        "p_value": p_value,
                        "same_distribution": p_value > 0.05,  # Common threshold
                        "dataset1_mean": dataset1[col].mean(),
                        "dataset2_mean": dataset2[col].mean(),
                        "dataset1_std": dataset1[col].std(),
                        "dataset2_std": dataset2[col].std(),
                    }
                except Exception as e:
                    results["feature_distribution"][col] = {
                        "error": str(e)
                    }
    
    # Print summary if verbose
    if verbose:
        print(f"Dataset Comparison Summary:")
        print(f"- Dataset 1 shape: {results['shape_comparison']['dataset1_shape']}")
        print(f"- Dataset 2 shape: {results['shape_comparison']['dataset2_shape']}")
        
        print("\nColumn comparison:")
        if results["column_comparison"]["same_columns"]:
            print("- Both datasets have identical columns")
        else:
            if results["column_comparison"]["only_in_dataset1"]:
                print(f"- Columns only in dataset 1: {results['column_comparison']['only_in_dataset1']}")
            if results["column_comparison"]["only_in_dataset2"]:
                print(f"- Columns only in dataset 2: {results['column_comparison']['only_in_dataset2']}")
        
        print("\nOverlap analysis:")
        if "overlap_count" in results["overlap_analysis"]:
            print(f"- {results['overlap_analysis']['overlap_count']} overlapping IDs found")
            print(f"- {results['overlap_analysis']['overlap_percentage']}% of dataset 1 IDs are in dataset 2")
            if results['overlap_analysis']['overlap_count'] > 0:
                warnings.warn(
                    f"Datasets have {results['overlap_analysis']['overlap_count']} overlapping IDs. "
                    f"This may lead to data leakage if one dataset was used for training."
                )
        elif "overlap_row_count" in results["overlap_analysis"]:
            print(f"- {results['overlap_analysis']['overlap_row_count']} overlapping rows found")
            print(f"- {results['overlap_analysis']['overlap_percentage']}% of dataset 1 rows are in dataset 2")
            if results['overlap_analysis']['overlap_row_count'] > 0:
                warnings.warn(
                    f"Datasets have {results['overlap_analysis']['overlap_row_count']} overlapping rows. "
                    f"This may lead to data leakage if one dataset was used for training."
                )
        else:
            print(f"- {results['overlap_analysis']['message']}")
            
        # Print feature distribution results if available
        if feature_distribution and "feature_distribution" in results:
            print("\nFeature distribution comparison:")
            for col, stats_data in results["feature_distribution"].items():
                if "error" in stats_data:
                    print(f"- {col}: Error in comparison - {stats_data['error']}")
                    continue
                    
                if stats_data["same_distribution"]:
                    print(f"- {col}: Similar distributions (p={stats_data['p_value']:.4f})")
                else:
                    print(f"- {col}: Different distributions (p={stats_data['p_value']:.4f})")
                    print(f"  Dataset 1: mean={stats_data['dataset1_mean']:.4f}, std={stats_data['dataset1_std']:.4f}")
                    print(f"  Dataset 2: mean={stats_data['dataset2_mean']:.4f}, std={stats_data['dataset2_std']:.4f}")
    
    return results

def check_data_leakage(
    report, 
    new_data,
    id_column=None, 
    threshold=0.05,
    verbose=True
):
    """
    Check if new data overlaps with training data in a report object.
    
    This utility helps ensure that when using report_metrics with new data (X_y syntax),
    the data is actually "new" and doesn't overlap significantly with training data.
    
    Parameters
    ----------
    report : EstimatorReport or ComparisonReport
        The report object containing reference to training data
    new_data : DataFrame or array
        New data to check for overlap with training data
    id_column : str, optional
        Column name to use for identifying unique rows
    threshold : float, default=0.05
        Maximum acceptable overlap percentage (0.05 = 5%)
    verbose : bool, default=True
        Whether to print comparison summary
        
    Returns
    -------
    bool
        True if significant overlap detected (exceeding threshold)
    dict
        Detailed leakage report from compare_datasets
        
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from skore import EstimatorReport
    >>> from skore.dataset_utils import check_data_leakage
    >>> 
    >>> # Create and train model
    >>> X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    >>> y_train = [0, 0, 1]
    >>> X_new = pd.DataFrame({"feature1": [2, 3, 4], "feature2": [5, 6, 7]})
    >>> y_new = [0, 1, 1]
    >>> 
    >>> model = RandomForestClassifier()
    >>> report = EstimatorReport(model, X_train=X_train, y_train=y_train)
    >>> 
    >>> # Check for data leakage before evaluating on "new" data
    >>> has_leakage, leakage_report = check_data_leakage(report, X_new)
    >>> 
    >>> if not has_leakage:
    >>>     report.report_metrics(X_y=(X_new, y_new))
    >>> else:
    >>>     print(f"Warning: {leakage_report['overlap_analysis']['overlap_percentage']}% overlap detected")
    """
    import warnings
    
    # Extract training data from report object
    if hasattr(report, "X_train") and report.X_train is not None:
        # EstimatorReport case
        train_data = report.X_train
    elif hasattr(report, "reports_") and len(report.reports_) > 0 and hasattr(report.reports_[0], "X_train"):
        # ComparisonReport case
        train_data = report.reports_[0].X_train  # Use first report's training data
    else:
        raise ValueError("Could not find training data in the provided report object")
    
    # Run comparison
    leakage_report = compare_datasets(train_data, new_data, id_column=id_column, verbose=verbose)
    
    # Determine if overlap exceeds threshold
    overlap_pct = 0
    if "overlap_percentage" in leakage_report["overlap_analysis"]:
        overlap_pct = leakage_report["overlap_analysis"]["overlap_percentage"]
    elif "overlap_row_count" in leakage_report["overlap_analysis"]:
        overlap_pct = leakage_report["overlap_analysis"]["overlap_percentage"]
        
    significant_overlap = overlap_pct > (threshold * 100)
    
    if significant_overlap and verbose:
        warnings.warn(
            f"Significant data overlap detected: {overlap_pct:.2f}% of training data "
            f"appears in evaluation data. This exceeds the threshold of {threshold*100:.1f}%. "
            f"This may lead to overly optimistic performance estimates.",
            UserWarning
        )
        
    return significant_overlap, leakage_report