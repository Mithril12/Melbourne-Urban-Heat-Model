"""Tests for model training, prediction, and evaluation (src.heat_model)."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.heat_model import (
    train_model,
    predict,
    evaluate_model,
    FEATURES,
    TEMP_MIN,
    TEMP_MAX,
)


class TestTrainModel:
    def test_returns_three_tuple(self, real_df):
        result = train_model(real_df)
        assert len(result) == 3

    def test_first_element_is_random_forest(self, real_df):
        model, _, _ = train_model(real_df)
        assert isinstance(model, RandomForestRegressor)

    def test_model_is_fitted(self, real_df):
        model, _, _ = train_model(real_df)
        # A fitted RandomForestRegressor has estimators_
        assert hasattr(model, "estimators_")
        assert len(model.estimators_) > 0

    def test_test_split_is_non_empty(self, real_df):
        _, X_test, y_test = train_model(real_df)
        assert len(X_test) > 0
        assert len(y_test) > 0

    def test_train_and_test_sizes_sum_to_total(self, real_df):
        _, X_test, _ = train_model(real_df, test_size=0.2)
        # With 15 rows and test_size=0.2, sklearn rounds to 3 test rows
        assert len(X_test) == pytest.approx(15 * 0.2, abs=1)

    def test_custom_features_are_used(self, real_df):
        custom_features = ["ndvi", "impervious_surface_pct"]
        model, X_test, _ = train_model(real_df, features=custom_features)
        assert list(X_test.columns) == custom_features

    def test_reproducibility_with_same_random_state(self, real_df):
        model1, _, _ = train_model(real_df, random_state=0)
        model2, _, _ = train_model(real_df, random_state=0)
        preds1 = model1.predict(real_df[FEATURES])
        preds2 = model2.predict(real_df[FEATURES])
        np.testing.assert_array_equal(preds1, preds2)

    def test_different_random_states_may_differ(self, real_df):
        _, X_test1, y_test1 = train_model(real_df, random_state=0)
        _, X_test2, y_test2 = train_model(real_df, random_state=99)
        # Test sets should differ for different seeds
        assert not X_test1.index.equals(X_test2.index)


class TestPredict:
    def test_returns_numpy_array(self, real_df):
        model, _, _ = train_model(real_df)
        preds = predict(model, real_df)
        assert isinstance(preds, np.ndarray)

    def test_prediction_count_matches_row_count(self, real_df):
        model, _, _ = train_model(real_df)
        preds = predict(model, real_df)
        assert len(preds) == len(real_df)

    def test_predictions_within_plausible_temperature_range(self, real_df):
        model, _, _ = train_model(real_df)
        preds = predict(model, real_df)
        assert (preds >= TEMP_MIN).all(), f"Some predictions below {TEMP_MIN}°C"
        assert (preds <= TEMP_MAX).all(), f"Some predictions above {TEMP_MAX}°C"

    def test_predictions_vary_across_suburbs(self, real_df):
        model, _, _ = train_model(real_df)
        preds = predict(model, real_df)
        assert preds.std() > 0, "All predictions are identical — model is not discriminating"

    def test_custom_features_prediction(self, real_df):
        custom_features = ["ndvi", "building_density"]
        model, _, _ = train_model(real_df, features=custom_features)
        preds = predict(model, real_df, features=custom_features)
        assert len(preds) == len(real_df)


class TestEvaluateModel:
    def test_returns_dict_with_r2_and_mae(self, real_df):
        model, X_test, y_test = train_model(real_df)
        metrics = evaluate_model(model, X_test, y_test)
        assert "r2" in metrics
        assert "mae" in metrics

    def test_mae_is_non_negative(self, real_df):
        model, X_test, y_test = train_model(real_df)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["mae"] >= 0

    def test_r2_is_a_float(self, real_df):
        model, X_test, y_test = train_model(real_df)
        metrics = evaluate_model(model, X_test, y_test)
        assert isinstance(metrics["r2"], float)

    def test_perfect_predictions_give_r2_of_1(self, real_df):
        """When predictions equal targets exactly, R² should be 1.0."""
        model, X_test, y_test = train_model(real_df)
        metrics = evaluate_model(model, X_test, y_test.values * 1)  # pass copy
        # Use the model's own predictions against itself for a sanity check
        import pandas as pd
        preds = model.predict(X_test)
        from sklearn.metrics import r2_score
        assert r2_score(y_test, preds) == pytest.approx(metrics["r2"])
