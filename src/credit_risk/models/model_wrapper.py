from datetime import datetime
import mlflow
import numpy as np
import pandas as pd
import tempfile
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from credit_risk.config import Tags
from loguru import logger
import os


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for model class."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the model."""
        # Get path from artifacts (accept both keys)
        raw = context.artifacts.get("credit-risk-model") or context.artifacts.get("risk_model")
        if not raw:
            raise RuntimeError(f"No bundled artifact named 'credit-risk-model' or 'risk_model'. "
                               f"Available: {list(context.artifacts.keys())}")

        # Normalize Windows backslashes to POSIX (serving runs on Linux)
        p = str(raw).replace("\\", "/")

        # Ensure we point to a flavor root (directory that contains MLmodel)
        def points_to_flavor_root(d: str) -> bool:
            return os.path.isfile(os.path.join(d, "MLmodel"))

        if not points_to_flavor_root(p):
            # Try one level deeper if the copy added an extra folder
            found = False
            if os.path.isdir(p):
                for child in os.listdir(p):
                    cand = os.path.join(p, child)
                    if points_to_flavor_root(cand):
                        p = cand
                        found = True
                        break
            if not found:
                # Try the canonical artifact-key location as a last resort
                canonical = "/model/artifacts/credit-risk-model"
                if points_to_flavor_root(canonical):
                    p = canonical
                else:
                    raise RuntimeError(
                        f"Bundled model flavor root not found. Tried: {raw} → {p} "
                        f"and {canonical}. Contents of {p!r}: {os.listdir(p) if os.path.isdir(p) else '(not a dir)'}"
                    )
                
        # Load using pyfunc (works regardless of underlying flavor)
        self.model = mlflow.pyfunc.load_model(p)

    
    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray):
        """Lazy-load model if necessary, then predict."""
        preds = self.model.predict(model_input)
        preds = preds.tolist() if hasattr(preds, "tolist") else preds
        return {"Credit risk prediction": ["bad" if int(p) == 1 else "good" for p in preds]}


    def log_register_model(
        self,
        wrapped_model_uri: str,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],
        input_example: pd.DataFrame,
    ):
        """
        Log and register the model.

        :param wrapped_model_uri: URI of the wrapped model
        :param pyfunc_model_name: Name of the PyFunc model
        :param experiment_name: Name of the experiment
        :param tags: Tags for the model
        :param code_paths: List of code paths
        :param input_example: Input example for the model
        """
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"model-wrapper-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            tmp_dir = tempfile.mkdtemp(prefix="wrapped_")
            local_wrapped = mlflow.artifacts.download_artifacts(
                artifact_uri=wrapped_model_uri,
                dst_path=tmp_dir
            )

            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"code/{whl_name}")
            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            signature = infer_signature(model_input=input_example, model_output={"Credit risk prediction": ["bad"]})
            model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={"credit-risk-model": local_wrapped},
                signature=signature,
                code_paths=code_paths,
                conda_env=conda_env,
            )

        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=pyfunc_model_name,
            tags=tags.to_dict(),
        )
        latest_version = registered_model.version
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=latest_version,
        )
        
        return latest_version