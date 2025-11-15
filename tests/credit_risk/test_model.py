import sys
from pathlib import Path
import types
import pandas as pd
import pytest
from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Add the src directory to the Python path if not already added
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import project modules after path setup
from credit_risk.models.model import Model
from credit_risk.models import model as model_mod


class DummyConfig:
    def __init__(self):
        self.numerical_features = ["num1"]
        self.categorical_features = ["cat1", "cat2"]
        self.target = "target"
        self.hyperparameters = {"n_estimators": 10}
        self.catalog_name = "catalog"
        self.schema_name = "schema"
        self.experiment_name_model = "exp_model"


class DummyTags:
    def to_dict(self):
        return {"tag": "value"}


def test_prepare_features_baseline_creates_logistic_and_expected_transformers():
    cfg = DummyConfig()
    tags = DummyTags()
    m = Model(config=cfg, tags=tags, spark=None, is_baseline_model=True)

    # create X_train with one categorical having >2 uniques and one with 2 uniques
    df = pd.DataFrame(
        {
            "num1": [0.1, 0.2, 0.3, 0.4],
            "cat1": ["a", "b", "c", "a"],  # >2 uniques
            "cat2": ["x", "y", "x", "y"],  # 2 uniques
            "target": [0, 1, 0, 1],
        }
    )
    m.X_train = df
    # prepare features should pick onehot for cat1 and ordinal for cat2
    m.prepare_features()

    assert isinstance(m.pipeline.named_steps["model"], LogisticRegression)
    preprocessor = m.pipeline.named_steps["preprocessor"]
    # transformer names should include num, cat_onehot and cat
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat_onehot" in transformer_names
    assert "cat" in transformer_names


def test_prepare_features_lgbm_creates_lgbm_and_custom_cat_transformer():
    cfg = DummyConfig()
    tags = DummyTags()
    m = Model(config=cfg, tags=tags, spark=None, is_baseline_model=False)

    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["a", "b", "a"],
            "cat2": ["x", "x", "y"],
            "target": [0, 1, 0],
        }
    )
    m.X_train = df
    m.prepare_features()

    assert isinstance(m.pipeline.named_steps["model"], LGBMClassifier)
    preprocessor = m.pipeline.named_steps["preprocessor"]
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert transformer_names == ["num", "cat"]
    # Access the inner pipeline for categorical transformer and ensure it has the custom encoder with cat_features
    cat_pipeline = preprocessor.transformers[1][1]
    # the pipeline should contain a step named 'lgbm_cat_encoder'
    assert "lgbm_cat_encoder" in cat_pipeline.named_steps
    encoder = cat_pipeline.named_steps["lgbm_cat_encoder"]
    assert getattr(encoder, "cat_features", None) == cfg.categorical_features


def test_get_hyperparameters_mlflow_returns_defaults_when_no_experiment(monkeypatch):
    cfg = DummyConfig()
    tags = DummyTags()
    m = Model(config=cfg, tags=tags, spark=None, is_baseline_model=False)

    class MockClientNoExp:
        def __init__(self):
            pass

        def get_experiment_by_name(self, name):
            return None

    monkeypatch.setattr(model_mod, "MlflowClient", MockClientNoExp)
    returned = m.get_hyperparameters_mlflow()
    assert returned == cfg.hyperparameters


def test_model_improved_true_and_false(monkeypatch):
    cfg = DummyConfig()
    tags = DummyTags()
    m = Model(config=cfg, tags=tags, spark=None, is_baseline_model=False)

    # Patch MlflowClient to return a simple object with model_id
    class MockClient:
        def __init__(self):
            pass

        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(model_id="model_123")

    monkeypatch.setattr(model_mod, "MlflowClient", MockClient)

    # Helper to patch mlflow.models.evaluate to return desired old metrics
    def make_eval(old_auc):
        def _evaluate(model_uri, eval_data, targets, model_type, evaluators):
            return SimpleNamespace(metrics={"precision_recall_auc": old_auc})
        return _evaluate

    # Case 1: current >= old -> True
    m.metrics = {"precision_recall_auc": 0.8}
    monkeypatch.setattr(model_mod.mlflow.models, "evaluate", make_eval(0.7))
    assert m.model_improved() is True

    # Case 2: current < old -> False
    m.metrics = {"precision_recall_auc": 0.5}
    monkeypatch.setattr(model_mod.mlflow.models, "evaluate", make_eval(0.6))
    assert m.model_improved() is False