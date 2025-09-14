import argparse
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from credit_risk.config import Config, Tags
from credit_risk.models.model import Model
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

# Tune hyperparameters for lightgbm model
lgbm_model.tune_hyperparameters()