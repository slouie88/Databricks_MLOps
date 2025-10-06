import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from credit_risk.config import Config, Tags
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from delta.tables import DeltaTable
import optuna


class Model:
    """A basic model class for Credit Risk Classification.

    This class handles data loading, feature preparation, model training, hyperparameter tuning, and MLflow logging.
    """

    def __init__(
        self, 
        config: Config, 
        tags: Tags, 
        spark: SparkSession,
        is_baseline_model: bool = False
    ) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        :param is_baseline_model: Flag indicating if this is a baseline model (Logistic Regression) or custom model (LightGBM)
        """
        self.config = config
        self.spark = spark
        self.is_baseline_model = is_baseline_model

        # Extract settings from the config
        self.numerical_features = self.config.numerical_features
        self.categorical_features = self.config.categorical_features
        self.target = self.config.target
        self.hyperparameters = self.config.hyperparameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_model
        self.model_name = f"{self.catalog_name}.{self.schema_name}.credit-risk-model"
        self.tags = tags.to_dict()


    def load_data(self) -> None:
        """Load training and testing data from Databricks Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("Loading data from Databricks Delta tables...")

        credit_risk_features = self.spark.table(f"{self.catalog_name}.{self.schema_name}.credit_risk_features").toPandas()

        self.train_set, self.test_set = train_test_split(credit_risk_features, test_size=0.2, random_state=42, stratify=credit_risk_features[self.target])
        self.train_set_spark = self.spark.createDataFrame(self.train_set)
        self.test_set_spark = self.spark.createDataFrame(self.test_set)

        self.X_train = self.train_set[self.numerical_features + self.categorical_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.numerical_features + self.categorical_features]
        self.y_test = self.test_set[self.target]
        self.eval_data = self.test_set[self.numerical_features + self.categorical_features + [self.target]]

        credit_risk_features_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.credit_risk_features")
        self.train_data_version = str(credit_risk_features_delta_table.history().select("version").first()[0])
        self.test_data_version = str(credit_risk_features_delta_table.history().select("version").first()[0])

        logger.info("Data successfully loaded.")


    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM classification model (or Logistic Regression for baseline model).
        """
        logger.info("Defining preprocessing pipeline...")
        
        # Define preprocessing pipeline for numerical features
        numerical_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]    # Scale numerical features for standardised scale.
        )

        # Define preprocessing pipeline for categorical features
        if self.is_baseline_model:
            onehot_transformer = Pipeline(
                steps=[("onehot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))] # One-hot encode to handle multi-class categorical features.
            )
            categorical_transformer = Pipeline(
                steps=[("ordinal_encoder", OrdinalEncoder())]   # Label encode to handle binary categorical features.
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, self.numerical_features),
                    ("cat_onehot", onehot_transformer, [col for col in self.categorical_features if self.X_train[col].nunique() > 2]),
                    ("cat", categorical_transformer, [col for col in self.categorical_features if self.X_train[col].nunique() == 2])
                ],
                remainder="drop"  # Drop any features not specified in the transformers.
            )
        else:
            class LGBMCategoricalTransformer(BaseEstimator, TransformerMixin):  # Define a custom transformer for LightGBM categorical encoding in a pipeline. Class defined within this method to keep model file self-contained.
                """Transformer that encodes categorical columns as integer codes for LightGBM.

                Unknown categories at transform time are encoded as -1. Encoding integers as category type for LightGBM's internal handling of categorical features.
                """

                def __init__(self, cat_features: list[str]) -> None:
                    """Initialize the transformer with categorical feature names."""
                    self.cat_features = cat_features
                    self.cat_maps_ = {}


                def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
                    """Fit the transformer to the DataFrame X."""
                    self.fit_transform(X)
                    return self


                def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
                    """Fit and transform the DataFrame X."""
                    X = X.copy()

                    for col in self.cat_features:
                        cat = pd.Categorical(X[col])
                        self.cat_maps_[col] = dict(zip(cat.categories, range(len(cat.categories)), strict=False))
                        X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")

                    return X


                def transform(self, X: pd.DataFrame) -> pd.DataFrame:
                    """Transform the DataFrame X by encoding categorical features as integers."""
                    X = X.copy()

                    for col in self.cat_features:
                        X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")

                    return X
                
            categorical_transformer = Pipeline(
                steps=[("lgbm_cat_encoder", LGBMCategoricalTransformer(cat_features=self.categorical_features))]   # Encode categorical features for LightGBM.
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, self.numerical_features),
                    ("cat", categorical_transformer, self.categorical_features)
                ]
            )

        # Define the overall model pipeline - use Logistic Regression as baseline classifier, LightGBM for otherwise
        self.model = LogisticRegression(max_iter=1000) if self.is_baseline_model else LGBMClassifier(n_jobs=-1, **self.hyperparameters)
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", self.model)])

        logger.info("Preprocessing pipeline successfully defined.")


    def tune_hyperparameters(self) -> None:
        """Tune hyperparameters using bayesian optimisation with cross-validation.

        Uses 5-fold cross-validation and TPE optimisation to find the best hyperparameters for the model.
        """
        logger.info("Starting hyperparameter tuning...")

        current_timestamp = pd.Timestamp.now(tz="Australia/Brisbane").strftime("%Y%m%d_%H%M%S")

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=f"hyperparameter_tuning_{current_timestamp}", tags=self.tags) as run:

            def objective(trial):
                with mlflow.start_run(nested=True):
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
                    }

                    model = LGBMClassifier(n_jobs=-1, **params)
                    pipeline = Pipeline(steps=[("preprocessor", self.pipeline.named_steps["preprocessor"]), ("model", model)])
                    scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=5, scoring="roc_auc")
                    cv_score = scores.mean()

                    mlflow.log_params(params)
                    mlflow.log_metric("roc_auc", cv_score)

                    return cv_score
                
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=100)

            best_params = study.best_params
            self.hyperparameters.update(best_params)
            self.model.set_params(**self.hyperparameters)
            self.pipeline.named_steps["model"] = self.model

            mlflow.log_params(best_params)
            mlflow.log_metric("best_roc_auc", study.best_value)

        logger.info(f"Hyperparameter tuning completed.")


    def get_hyperparameters_mlflow(self) -> dict:
        """Fetch the best hyperparameters from the latest MLflow hyperparameter tuning run.

        If no mlflow experiments or hyperparameter tuning runs are found, returns the default hyperparameters from the config.

        :return best_params: Dictionary of best hyperparameters
        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            logger.info(f"Experiment '{self.experiment_name}' does not exist. Returning default hyperparameters from config.")
            return self.hyperparameters

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], 
            filter_string="tags.mlflow.runName LIKE 'hyperparameter_tuning_%'", 
            order_by=["metrics.best_roc_auc DESC"], 
            max_results=1
        )
        if not runs:
            logger.info(f"No hyperparameter tuning runs found in experiment '{self.experiment_name}'. Returning default hyperparameters from config.")
            return self.hyperparameters

        best_run = runs[0]
        best_params = {key: value for key, value in best_run.data.params.items()}

        logger.info(f"Best hyperparameters fetched from MLflow run ID {best_run.info.run_id}: {best_params}")

        return best_params


    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training...")

        self.hyperparameters = self.get_hyperparameters_mlflow() if not self.is_baseline_model else self.hyperparameters
        self.pipeline.fit(self.X_train, self.y_train)

        logger.info("Training completed.")


    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))
            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.credit_risk_features",
                version=self.train_data_version,
            )
            mlflow.log_input(train_dataset, context="training")
            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.credit_risk_features",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")
            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="credit-risk-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )

            if not self.is_baseline_model:
                mlflow.log_params(self.hyperparameters)

            eval_data = self.X_test.copy()
            eval_data[self.config.target] = self.y_test
            result = mlflow.models.evaluate(
                self.model_info.model_uri,
                eval_data,
                targets=self.config.target,
                model_type="classifier",
                evaluators=["default"],
            )
            self.metrics = result.metrics


    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("Registering the model in UC...")

        model_uri = f"runs:/{self.run_id}/credit-risk-model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
    

    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using ROC AUC.

        :return: True if the current model performs better, False otherwise.
        """
        client = MlflowClient()
        latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
        latest_model_uri = f"models:/{latest_model_version.model_id}"

        result = mlflow.models.evaluate(
            latest_model_uri,
            self.eval_data,
            targets=self.config.target,
            model_type="classifier",
            evaluators=["default"],
        )
        metrics_old = result.metrics
        if self.metrics["precision_recall_auc"] >= metrics_old["precision_recall_auc"]:
            logger.info("Current model performs better. Returning True.")
            return True
        else:
            logger.info("Current model does not improve over latest. Returning False.")
            return False