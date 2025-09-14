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
from credit_risk.config import Config, Tags
from credit_risk.models.model import Model
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
# MAGIC ### Tune LightGBM Model Hyperparameters

# COMMAND ----------

# Initialize lightgbm classification model
lgbm_model = Model(config=config, tags=tags, spark=spark)
logger.info("model initialized.")

# Load credit risk data
lgbm_model.load_data()
logger.info("data loaded.")

# Prepare credit risk features
lgbm_model.prepare_features()

# Tune hyperparameters for lightgbm model
lgbm_model.tune_hyperparameters()