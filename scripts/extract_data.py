import argparse
import yaml
from loguru import logger
from pyspark.sql import SparkSession
import pandas as pd
from credit_risk.config import Config
from credit_risk.data_extractor import DataExtractor


# Configuration for project
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

args = parser.parse_args()
project_config_yml_filename = "project_config_credit_risk.yml"  # Placeholder this with target project_config yaml file. If multiple projects in src, then pass the appropriate projdect_config yaml file.
config_path = f"{args.root_path}/files/{project_config_yml_filename}"
config = Config.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the source csv ataset
spark = SparkSession.builder.getOrCreate()
data_filepath = f"{args.root_path}/files/data/german_credit_data.csv"
pd_df = pd.read_csv(data_filepath)
logger.info("Source csv data loaded for processing.")

# Initialize DataExtractor
data_extractor = DataExtractor(config, spark)

# Preprocess the data
data_extractor.extract_to_feature_table(pd_df)
logger.info("Data extracted to Unity Catalog feature table.")
