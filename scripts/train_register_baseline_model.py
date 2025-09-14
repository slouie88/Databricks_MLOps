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

# Initialize baseline classification model
baseline_model = Model(
    config=config, 
    tags=tags, 
    spark=spark,
    is_baseline_model=True
)
logger.info("baseline model initialized.")

# Load credit risk data
baseline_model.load_data()
logger.info("baseline data loaded.")

# Prepare credit risk features
baseline_model.prepare_features()

# Train the baseline model
baseline_model.train()
logger.info("baseline model training completed.")

# Log the baseline model
baseline_model.log_model()

# Register the model
baseline_model.register_model()
databricks_mlops_v = version("credit_risk")

pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.baseline_model"
code_paths=[f"{root_path}/artifacts/.internal/credit_risk-{databricks_mlops_v}-py3-none-any.whl"]

wrapper = ModelWrapper()
latest_version = wrapper.log_register_model(wrapped_model_uri=f"{baseline_model.model_info.model_uri}",
                        pyfunc_model_name=pyfunc_model_name,
                        experiment_name=config.experiment_name_pyfunc_model,
                        input_example=baseline_model.X_test[0:1],
                        tags=tags,
                        code_paths=code_paths)

logger.info("New model registered with version:", latest_version)
dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
dbutils.jobs.taskValues.set(key="model_updated", value=1)
