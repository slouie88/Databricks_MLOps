# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Packages

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession
from importlib.metadata import version
from credit_risk.config import Config, Tags
from credit_risk.models.model import Model
from credit_risk.models.model_wrapper import ModelWrapper
from dotenv import load_dotenv
import mlflow
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration

# COMMAND ----------
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()

config_path = "../project_config_credit_risk.yml"
config = Config.from_yaml(config_path=config_path, env="dev")
tags_dict = {"git_sha": "abcd12345", "branch": "main"}
tags = Tags(**tags_dict)
print(config)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Train, log, and register baseline model

# COMMAND ----------

# Initialize lightgbm classification model
lgbm_model = Model(config=config, tags=tags, spark=spark)
logger.info("model initialized.")

# Load credit risk data
lgbm_model.load_data()
logger.info("data loaded.")

# Prepare credit risk features
lgbm_model.prepare_features()

# Train the lightgbm model
lgbm_model.train()
logger.info("model training completed.")

# Log the lightgbm model
lgbm_model.log_model()

# Evaluate lightgbm model
model_improved = lgbm_model.model_improved()
logger.info("model evaluation completed, model improved: %s", model_improved)

if model_improved:
    # Register the model
    lgbm_model.register_model()
    databricks_mlops_v = version("credit_risk") # If running locally, ensure you build the credit_risk package first in the project root directory.
    logger.info(f"Wrapped model uri: {lgbm_model.model_info.model_uri}")

    pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.pyfunc_credit_risk_model"
    code_paths=[f"../dist/credit_risk-{databricks_mlops_v}-py3-none-any.whl"]

    wrapper = ModelWrapper()
    latest_version = wrapper.log_register_model(wrapped_model_uri=f"{lgbm_model.model_info.model_uri}",
                            pyfunc_model_name=pyfunc_model_name,
                            experiment_name=config.experiment_name_pyfunc_model,
                            input_example=lgbm_model.X_test[0:1],
                            tags=tags,
                            code_paths=code_paths)

    logger.info("New model registered with version:", latest_version)
