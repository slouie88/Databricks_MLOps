import mlflow
import pandas as pd
from delta.tables import DeltaTable
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from credit_risk.config import Config, Tags
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class BasicModel:
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
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.credit_risk_model_basic"
        self.tags = tags.to_dict()


    def load_data(self) -> None:
        """Load training and testing data from Databricks Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks Delta tables...")

        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set_spark.toPandas()

        self.X_train = self.train_set[self.numerical_features + self.categorical_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.numerical_features + self.categorical_features]
        self.y_test = self.test_set[self.target]
        self.eval_data = self.test_set[self.numerical_features + self.categorical_features + [self.target]]

        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])

        logger.info("✅ Data successfully loaded.")


    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM classification model (or Logistic Regression for baseline model).
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        
        # Define preprocessing pipeline for numerical and categorical features
        numerical_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]    # Scale numerical features for standardised scale.
        )
        onehot_transformer = Pipeline(
            steps=[("onehot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))] # One-hot encode to handle multi-class categorical features.
        )
        categorical_transformer = Pipeline(
            steps=[("label_encoder", LabelEncoder())]   # Label encode to handle binary categorical features.
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat_onehot", onehot_transformer, [col for col in self.categorical_features if self.X_train[col].nunique() > 2]),
                ("cat_label", categorical_transformer, [col for col in self.categorical_features if self.X_train[col].nunique() == 2])
            ],
            remainder="drop"  # Drop any features not specified in the transformers.
        )

        # Define the overall model pipeline. Use Logistic Regression as baseline classifier, LightGBM for otherwise.
        self.model = LogisticRegression(max_iter=1000) if self.is_baseline_model else LGBMClassifier(**self.hyperparameters)
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", self.model)])

        logger.info("✅ Preprocessing pipeline successfully defined.")


    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")

        self.pipeline.fit(self.X_train, self.y_train)

        logger.info("✅ Training completed.")


    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))
            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.train_data_version,
            )
            mlflow.log_input(train_dataset, context="training")
            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.test_set",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")
            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="credit-risk-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )
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
        logger.info("🔄 Registering the model in UC...")

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/credit-risk-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

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
        if self.metrics["roc_auc"] >= metrics_old["roc_auc"]:
            logger.info("Current model performs better. Returning True.")
            return True
        else:
            logger.info("Current model does not improve over latest. Returning False.")
            return False