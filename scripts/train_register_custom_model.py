import argparse
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from importlib.metadata import version
from credit_risk.config import Config, Tags
from credit_risk.models.model import Model
from credit_risk.models.model_wrapper import ModelWrapper
from dotenv import load_dotenv
import mlflow
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
parser.add_argument("--job_run_id", type=str, required=True, help="run id of the databricks job")
parser.add_argument("--branch", type=str, required=True, help="branch of the project")

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config_credit_risk.yml"

# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load project configuration
config = Config.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

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
    databricks_mlops_v = version("credit_risk")

    pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.pyfunc_credit_risk_model"
    code_paths=[f"{root_path}/artifacts/.internal/credit_risk-{databricks_mlops_v}-py3-none-any.whl"]

    wrapper = ModelWrapper()
    latest_version = wrapper.log_register_model(wrapped_model_uri=f"{lgbm_model.model_info.model_uri}",
                            pyfunc_model_name=pyfunc_model_name,
                            experiment_name=config.experiment_name_pyfunc_model,
                            input_example=lgbm_model.X_test[0:1],
                            tags=tags,
                            code_paths=code_paths)

    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
