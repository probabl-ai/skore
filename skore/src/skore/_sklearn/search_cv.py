import numpy as np
from sklearn.model_selection._search import BaseSearchCV, _fit_and_score
from sklearn.model_selection._validation import _aggregate_score_dicts


def _monkey_patch_fit_and_score(original_fit_and_score):
    def patched_fit_and_score(
        estimator,
        X,
        y,
        *,
        scorer,
        train,
        test,
        verbose,
        parameters,
        fit_params,
        score_params,
        return_train_score=False,
        return_parameters=False,
        return_n_test_samples=False,
        return_times=False,
        return_estimator=False,
        split_progress=None,
        candidate_progress=None,
        error_score=np.nan,
    ):
        results = original_fit_and_score(
            estimator,
            X,
            y,
            scorer=scorer,
            train=train,
            test=test,
            verbose=verbose,
            parameters=parameters,
            fit_params=fit_params,
            score_params=score_params,
            return_train_score=return_train_score,
            return_parameters=return_parameters,
            return_n_test_samples=return_n_test_samples,
            return_times=return_times,
            return_estimator=True,
            split_progress=split_progress,
            candidate_progress=candidate_progress,
            error_score=error_score,
        )
        results.update({"train_indices": train, "test_indices": test})
        return results

    return patched_fit_and_score


def _monkey_patch_format_results(original_format_results):
    def patched_format_results(
        self, candidate_params, n_splits, out, more_results=None
    ):
        extra_results = [
            {
                "estimator": item["estimator"],
                "train_indices": item["train_indices"],
                "test_indices": item["test_indices"],
            }
            for item in out
        ]
        result = original_format_results(
            self, candidate_params, n_splits, out, more_results
        )
        result.update(_aggregate_score_dicts(extra_results))
        return result

    return patched_format_results


patched_fit_and_score = _monkey_patch_fit_and_score(_fit_and_score)
patched_format_results = _monkey_patch_format_results(BaseSearchCV._format_results)
